"""
Run the accessibility model for a study area.

    python -m ikob2.cli.accessibility --data-root <root> \
        --study utrecht_nl --run s0 --modes car bike pt --ownership

With --data-root every input and the output folder come from the data
folder layout; each can also be given explicitly (--kwb, --skims,
--sector-jobs, --out, ...).

Inputs it combines (see docs/pipeline.md): the skim store (origins are the
study buurten, destinations all buurten), the 44 household-type x income
segments (computed on the fly from KWB and the StatLine snapshots), imputed
sector jobs (`cli.segments jobs`), the reference budgets, the Weibull time
margins, the car cost model, PT fares and shared-bicycle tariffs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ikob2 import params as params_mod
from ikob2.domain.filter_config import INDEPENDENCE, CopulaSpec, CurveSpec
from ikob2.run.accessibility import ModeMatrices, run_accessibility
from ikob2.segments import statline
from ikob2.segments.bridge import envelope_segment_names, load_reference_budgets
from ikob2.segments.config import SegmentConfig
from ikob2.segments.lisa import sector_wages
from ikob2.segments.pipeline import run_pipeline
from ikob2.segments.time_margins import load_time_margins
from ikob2.segments.wfh import (
    lisa_sector_wfh_share,
    sbi_wfh_share,
    sector_education_mix,
    wfh_incidence_by_education,
)
from ikob2.skims.car import (
    CarCostModel,
    DetourModel,
    car_time_and_cost,
    crowfly_km,
)
from ikob2.skims.pt_fare import PtFareModel
from ikob2.skims.store import SkimStore
from ikob2.utils.paths import DataLayout, resolve_input

logger = logging.getLogger("ikob2.cli.accessibility")

# command-line flag -> parameter it overrides (flags default to None)
FLAGS = {
    "kwb_year": "accessibility.kwb_year", "jobs_year": "accessibility.jobs_year",
    "budgets": "paths.budgets", "modes": "accessibility.modes",
    "time_shape": "accessibility.time_shape", "cutoff": "accessibility.cutoff",
    "exp_calibration": "accessibility.exp_calibration",
    "cost_gate": "accessibility.cost_gate",
    "legs_per_tour": "accessibility.legs_per_tour",
    "censored": "accessibility.censored",
    "population_basis": "accessibility.population_basis",
    "copula": "accessibility.copula", "theta": "accessibility.theta",
    "spec": "accessibility.spec", "epsilon": "accessibility.epsilon",
    "wage_period": "accessibility.wage_period",
    "wfh_period": "accessibility.wfh_period",
    "ownership": "accessibility.ownership",
    "common_jobs": "accessibility.common_jobs",
    "shared_bike": "accessibility.shared_bike",
    "bike_fixed_min": "bike_leg.fixed_minutes",
    "ovfiets_eur": "shared_bike.ovfiets_eur",
    "dockless_unlock_eur": "shared_bike.dockless_unlock_eur",
    "dockless_per_min_eur": "shared_bike.dockless_per_min_eur",
    "dockless_model": "shared_bike.dockless_model",
    "lime_scale": "shared_bike.lime_scale",
    "flat_eur": "shared_bike.flat_eur",
    "flat_method": "shared_bike.flat_method",
    "report_usage": "shared_bike.report_usage",
    "price_scales": "paths.lime_price_scales",
    "pt_fare_scales": "paths.pt_fare_scales",
    "car_model": "car.default_model", "parking_search": "car.parking_search",
    "pt_rail_table": "pt_fare.rail_table",
    "pt_rail_discount": "pt_fare.rail_discount",
    "pt_rail_anchors": "pt_fare.rail_anchors",
    "pt_rail_1km": "pt_fare.rail_eur_per_km_at_1km",
    "pt_rail_100km": "pt_fare.rail_eur_per_km_at_100km",
    "pt_regional_boarding": "pt_fare.regional_boarding_eur",
    "pt_regional_km": "pt_fare.regional_eur_per_km",
    "pt_boardings": "pt_fare.boardings",
}


def _sector_tables(statline_dir: Path, wage_period: str, wfh_period: str):
    wage_raw = pd.read_csv(
        statline.snapshot_path(statline_dir, statline.WAGE_SNAPSHOT,
                               statline.WAGE_TABLE, wage_period),
        dtype={"BedrijfstakkenBranchesSBI2008": str})
    hw = pd.read_csv(statline.snapshot_path(
        statline_dir, statline.HOME_WORK_SNAPSHOT, statline.HOME_WORK_TABLE,
        wfh_period), dtype=str)
    hw["WerkzameBeroepsbevolking_1"] = pd.to_numeric(
        hw["WerkzameBeroepsbevolking_1"])
    se = pd.read_csv(statline.snapshot_path(
        statline_dir, statline.SECTOR_EDUCATION_SNAPSHOT,
        statline.SECTOR_EDUCATION_TABLE, "2010JJ00"),
        dtype={"Onderwijsniveau": str, "BedrijfstakkenSBI2008": str})
    sbi = sbi_wfh_share(wfh_incidence_by_education(hw),
                        sector_education_mix(se))
    jobs = wage_raw.assign(k=wage_raw["BedrijfstakkenBranchesSBI2008"]
                           .str.strip()).set_index("k")["Banen_1"]
    return sector_wages(wage_raw), lisa_sector_wfh_share(sbi, jobs)


def build_matrices(store, zones, modes, *, detour, car_model, parking_search,
                   distance_store=None, pt_fare_model=None,
                   parking_arrival_min=None, departure_factor=None):
    """Mode -> ModeMatrices over (store origins, all zones).

    distance_store : optional second store holding the car `distance`
        variable (same origins and near/far layers), so that a scenario
        store with different travel times (e.g. peak load) can reuse the
        routed distances of the base store.
    """
    dstore = distance_store or store
    if distance_store is not None:
        for layer in ("near", "far"):
            if (distance_store.origins != store.origins
                    or distance_store.layer(layer).destinations
                    != store.layer(layer).destinations):
                raise ValueError(f"distance store and time store differ in "
                                 f"origins or the '{layer}' layer.")
    codes = [str(c) for c in zones.codes]
    idx = {c: i for i, c in enumerate(codes)}
    o_idx = np.array([idx[o] for o in store.origins])
    xy = np.column_stack([zones.centroid_x, zones.centroid_y])
    urb = np.asarray(zones.attributes.get(
        "stedelijkheid_adressen_per_km2", np.full(len(codes), np.nan)))
    out = {}
    for mode in modes:
        if mode not in ("car", "bike", "pt"):
            raise SystemExit(f"Mode '{mode}' has no time margin (margins "
                             f"exist for bike, pt, car).")
        if mode == "pt":
            # PT is computed at buurt level for every destination
            t = store.block("all", "pt", "time", destinations=codes)
            if ("all", "pt", "rail_km") in store.arrays():
                model = pt_fare_model or PtFareModel()
                fare = model.fare(
                    store.block("all", "pt", "rail_km", destinations=codes),
                    store.block("all", "pt", "other_km", destinations=codes),
                    store.block("all", "pt", "other_boardings",
                                destinations=codes))
                rail = store.block("all", "pt", "rail_km", destinations=codes)
                other = store.block("all", "pt", "other_km", destinations=codes)
                tot = rail + other
                share = np.where(tot > 0, rail / np.where(tot > 0, tot, 1.0),
                                 1.0).astype(np.float32)
                out[mode] = ModeMatrices(t, fare, model.matrix_id,
                                         rail_share=share, fare=fare)
            else:
                logger.warning("PT store has no fare inputs (rebuild with "
                               "build-pt): time-only gate.")
                out[mode] = ModeMatrices(t)
            continue
        t = store.combined(mode, "time", codes, near="near", far="far")
        if mode == "car":
            if ("near", "car", "distance") in dstore.arrays():
                # routed distances (local Valhalla) where available
                dist = dstore.combined("car", "distance", codes, near="near",
                                       far="far")
                fallback = detour.route_km(crowfly_km(xy[o_idx], xy))
                dist = np.where(np.isfinite(dist), dist, fallback)
            else:
                dist = detour.route_km(crowfly_km(xy[o_idx], xy))
            time, cost = car_time_and_cost(
                t, dist, car_model,
                origin_urbanisation=urb[o_idx] if parking_search else None,
                dest_urbanisation=urb if parking_search else None,
                parking_arrival_min=parking_arrival_min,
                departure_factor=departure_factor)
            out[mode] = ModeMatrices(time, cost, car_model.matrix_id)
        elif mode == "bike":
            out[mode] = ModeMatrices(t)
    return out, codes


def pt_fare_model(prm) -> PtFareModel:
    """Fare model from the `pt_fare` parameters (rail anchors, regional
    charge, optional rail tariff table CSV with columns km, eur)."""
    return PtFareModel.from_params(prm.pt_fare)


def time_curve(shape: str, cutoff: float, calibration: str = "mean") -> CurveSpec:
    """A common time margin for the impedance-shape comparison: 'step' is
    the hard cut-off at `cutoff` minutes (isochrone); 'exponential' is
    calibrated to it: 'mean' gives the exponential the same mean
    acceptable time (rate 1/cutoff, the paper's moment matching), 'half'
    makes acceptance 50% at the cut-off (rate ln 2 / cutoff)."""
    if shape == "step":
        return CurveSpec("step", (float(cutoff),))
    if shape == "exponential":
        rate = (1.0 / cutoff if calibration == "mean"
                else float(np.log(2.0) / cutoff))
        return CurveSpec("exponential", (rate,))
    raise ValueError(f"Unknown time shape {shape!r}.")


def codes_all(zones) -> list[str]:
    return [str(c) for c in zones.codes]


def tariffs_from(prm):
    """SharedBikeTariffs of the `shared_bike` parameters."""
    from ikob2.run.shared_bike import SharedBikeTariffs

    s = prm.shared_bike
    return SharedBikeTariffs(
        s.ovfiets_eur, s.dockless_unlock_eur, s.dockless_per_min_eur,
        tuple(tuple(t) for t in s.lime_tiers), tuple(s.hub_tariffs),
        s.dockless_model, s.lime_scale, s.flat_eur)


def load_chains(prm, args, store, codes, model):
    """(chains, fares, bicycle share per origin) of the shared-bicycle
    variants: the store modes pt, pt_bw and the egress modes."""
    from ikob2.segments.ownership import load_bike_ownership

    chains, fares = {}, {}
    # egress chains per hub kind (pt_wb_<kind>, pt_bb_<kind>); with no kinds
    # in shared_bike.egress_hub_kinds the plain pt_wb / pt_bb are used
    kinds = list(prm.shared_bike.egress_hub_kinds)
    suffix = prm.shared_bike.egress_suffix.to_dict()
    egress = [f"{m}_{k}" for k in kinds for m in ("pt_wb", "pt_bb")] \
        or ["pt_wb", "pt_bb"]
    for mode in ("pt", "pt_bw", *egress):
        # a kind may read other skim modes (S2: pt_wb_lime_s2 with more hubs)
        stored = mode + suffix.get(mode.split("_", 2)[-1], "") \
            if mode.startswith(("pt_wb_", "pt_bb_")) else mode
        if ("all", stored, "time") not in store.arrays():
            raise SystemExit(f"Store lacks mode '{stored}': build it with "
                             f"`cli.skims build-pt --mode-name {stored}` "
                             f"(hub kinds: shared_bike.egress_hub_kinds).")
        def blk(v, stored=stored):
            return store.block("all", stored, v, destinations=codes)
        chains[mode] = {"time": blk("time")}
        if mode.startswith("pt_bw") or mode.startswith("pt_bb"):
            chains[mode]["access_min"] = blk("access_min")
        if mode.startswith(("pt_wb", "pt_bb")) \
                and ("all", stored, "egress_min") in store.arrays():
            chains[mode]["egress_min"] = blk("egress_min")
        chains[mode]["fare"] = model.fare(blk("rail_km"), blk("other_km"),
                                          blk("other_boardings"))
        rail, other = blk("rail_km"), blk("other_km")
        tot = rail + other
        chains[mode]["rail_share"] = np.where(
            tot > 0, rail / np.where(tot > 0, tot, 1.0), 1.0).astype(np.float32)
        fares[mode] = chains[mode]["fare"]
    share = load_bike_ownership(args.bike_ownership,
                                store.origins).to_numpy()
    return chains, fares, share


def shared_bike_matrices(prm, chains, fares, share, matrices, tariffs=None):
    """The requested shared-bicycle variants as chain modes."""
    from ikob2.run.shared_bike import dockless_mode, shared_bike_modes

    tariffs = tariffs or tariffs_from(prm)
    fixed = prm.bike_leg.fixed_minutes
    variants = [v for v in prm.accessibility.shared_bike if v != "v4"]
    out = shared_bike_modes(chains, fares, share, tariffs, variants=variants,
                            bike_fixed_min=fixed)
    if "v4" in prm.accessibility.shared_bike:
        if "bike" not in matrices:
            raise SystemExit("v4 needs the bicycle mode: add 'bike' to "
                             "--modes.")
        out["bike_v4"] = dockless_mode(matrices["bike"].time, share, tariffs,
                                       fixed)
    return out


def _pt_cost(prm, args, store, pop, seg_names, fare_scale):
    """Public cost per year of the fare concession at baseline volume, from
    the ODiN fare spending by income decile (`cli.segments pt-spend`)."""
    from ikob2.run.costs import pt_fare_cost

    root = (args.data_root or os.environ.get(params_mod.ENV_ROOT)
            or prm.paths.data_root)
    f = DataLayout(Path(root)).pt_spend() if root else None
    if f is None or not f.exists():
        logger.warning("No PT fare spending table (%s): run `cli.segments "
                       "pt-spend`; the cost of the concession is not reported.", f)
        return {}
    pop_o = (pop.set_index("buurtcode") if "buurtcode" in pop.columns
             else pop).reindex(store.origins)
    persons = pop_o[[n for n in seg_names if n in pop_o.columns]].sum().to_dict()
    cost = pt_fare_cost(pd.read_csv(f), persons, fare_scale)
    print(f"PT fare concession: public cost EUR {cost['eur_year']:,.0f} per year "
          f"(low estimate EUR {cost['eur_year_national']:,.0f})")
    return cost


def _hubs_for_export(prm, args):
    """The hub locations of the shared-bicycle chains, for the run products
    (None when the hub files cannot be found)."""
    from ikob2.skims import hubs as hubs_mod

    root = args.data_root or os.environ.get(params_mod.ENV_ROOT) \
        or prm.paths.data_root
    try:
        return hubs_mod.load_hubs(
            prm.pt.hub_files, hubs_mod.search_dirs(root),
            kinds=prm.pt.hub_kinds, tariffs=prm.shared_bike.hub_tariffs)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Hubs not exported: %s", exc)
        return None


def parse_vot(items) -> dict[str, float]:
    """['car=10', 'pt=9'] -> {'car': 10.0, 'pt': 9.0} (EUR per hour)."""
    out = {}
    for item in items or []:
        mode, _, value = item.partition("=")
        out[mode.strip()] = float(value)
    return out


def resolve(args) -> params_mod.Params:
    """Parameters of the run (file, --set, flags, --vot); the flag
    attributes of `args` are filled with the resolved values."""
    prm = params_mod.from_args(args, FLAGS)
    if args.vot:
        prm = prm.with_values({f"vot.{m}": v
                               for m, v in parse_vot(args.vot).items()})
    for name, key in FLAGS.items():
        setattr(args, name, prm.get(key))
    return prm


def resolve_paths(args, prm) -> None:
    """Fill unset paths from the data folder layout (--data-root, else
    $IKOB_DATA_ROOT, else paths.data_root). Reference files named in the
    parameters (budgets, margins, tariff tables) are used as given when that
    path exists, else looked up in their folder of the layout. Without a
    data root every path must be given explicitly."""
    root = (args.data_root or os.environ.get(params_mod.ENV_ROOT)
            or prm.paths.data_root)
    lay = DataLayout(Path(root)) if root else None
    if lay is not None:
        args.kwb = args.kwb or str(lay.kwb(args.kwb_year,
                                           prm.paths.kwb_version))
        args.skims = args.skims or str(lay.skim_dir(args.study))
        args.sector_jobs = args.sector_jobs or str(
            lay.sector_jobs(args.jobs_year))
        args.out = args.out or str(lay.run_dir(args.run))
        args.statline = args.statline or str(lay.statline())
        args.bike_ownership = args.bike_ownership or str(lay.bike_ownership())
        args.car_availability = args.car_availability or str(
            lay.car_availability())
        if not args.detour and lay.detour_model().exists():
            args.detour = str(lay.detour_model())
        args.margins = args.margins or str(lay.survey_margins())
    folder = (lambda sub: lay.inputs / sub) if lay is not None else (
        lambda sub: None)
    args.budgets = str(resolve_input(args.budgets, folder("envelope")))
    if args.margins:
        args.margins = str(resolve_input(args.margins, folder("survey")))
    for name in ("price_scales", "pt_fare_scales"):
        if getattr(args, name):
            setattr(args, name, str(resolve_input(getattr(args, name),
                                                  folder("tariffs"))))
    missing = [n for n in ("kwb", "skims", "sector_jobs", "out", "statline",
                           "margins") if not getattr(args, n)]
    if missing:
        raise SystemExit(f"Give --{', --'.join(m.replace('_', '-') for m in missing)}"
                         f" or --data-root (with --study and --run).")
    absent = [f"{flag} {getattr(args, n)}" for n, flag in (
        ("budgets", "--budgets"), ("margins", "--margins"),
        ("price_scales", "--price-scales"),
        ("pt_fare_scales", "--pt-fare-scales"))
        if getattr(args, n) and not Path(getattr(args, n)).exists()]
    if absent:
        raise SystemExit("Reference file(s) not found: " + "; ".join(absent)
                         + ". Seed the data folder (python -m "
                           "ikob2.cli.layout create) or give the path.")


def input_fingerprints(args) -> dict:
    """Path and SHA-256 of every input file of the run, so an archived data
    folder can be matched to run.json."""
    import hashlib

    files = {n: getattr(args, n, None) for n in (
        "budgets", "margins", "price_scales", "pt_fare_scales", "sector_jobs",
        "bike_ownership", "car_availability", "detour", "kwb")}
    if args.statline:
        for f in sorted(Path(args.statline).glob("*.csv")):
            files[f"statline/{f.name}"] = f
    out = {}
    for name, path in files.items():
        if not path or not Path(path).is_file():
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out[name] = {"path": str(Path(path).resolve()), "sha256": h.hexdigest()}
    return out


def cmd_run(args) -> None:
    from ikob2.data.geopackage import load_cbs_buurten

    prm = resolve(args)
    resolve_paths(args, prm)

    logging.getLogger("ikob2.data.geopackage").setLevel(logging.ERROR)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zones, _ = load_cbs_buurten(args.kwb)
    store = SkimStore.open(args.skims)
    dist_store = None
    if args.distance_study:
        dist_store = SkimStore.open(
            Path(args.skims).parent / args.distance_study)
    detour = (DetourModel.load(args.detour) if args.detour
              else DetourModel.constant(prm.car.detour_constant))
    if not args.detour:
        logger.warning("No calibrated detour model: using a constant %g "
                       "route/crow-fly factor for car distances.",
                       prm.car.detour_constant)
    matrices, dest_codes = build_matrices(
        store, zones, args.modes, detour=detour,
        car_model=CarCostModel.from_params(prm.car.models.get(args.car_model)),
        parking_search=args.parking_search,
        distance_store=dist_store, pt_fare_model=pt_fare_model(prm),
        parking_arrival_min=prm.car.parking_arrival_min.to_dict(),
        departure_factor=prm.car.departure_factor)

    seg_cfg = SegmentConfig()
    segs = run_pipeline(args.kwb, args.statline, seg_cfg)
    pop = (segs.population_scaled if args.population_basis
           == "population_scaled" else segs.household_based)

    sector_jobs = pd.read_csv(args.sector_jobs, index_col=0)
    wage, wfh = _sector_tables(Path(args.statline), args.wage_period,
                               args.wfh_period)
    envelope = load_reference_budgets(args.budgets,
                                      censored=args.censored,
                                      legs_per_tour=args.legs_per_tour)
    margins = load_time_margins(args.margins)
    envelope_arg = envelope
    if args.time_shape != "weibull":
        margins = {(m, w): time_curve(args.time_shape, args.cutoff,
                                      args.exp_calibration)
                   for m in args.modes for w in ("no_wfh", "wfh_possible")}
    if not args.cost_gate:
        envelope_arg = None
    copula = (INDEPENDENCE if args.copula == "independence"
              else CopulaSpec(args.copula, args.theta
                              if args.copula in ("gumbel", "frank") else None))

    seg_names = envelope_segment_names(envelope)
    price_scale = None
    if args.price_scales:
        from ikob2.run.shared_bike import (load_price_scales,
                                           segment_price_scales)
        price_scale = segment_price_scales(seg_names, load_price_scales(
            args.price_scales))
        if any(v != 1.0 for v in price_scale.values()):
            logger.info("Lime price scales differ from 1 for %d segments",
                        sum(v != 1.0 for v in price_scale.values()))
    fare_scale = None
    if args.pt_fare_scales:
        from ikob2.run.shared_bike import (load_price_scales,
                                           segment_price_scales)
        fare_scale = segment_price_scales(seg_names, load_price_scales(
            args.pt_fare_scales))
    scenario = {}
    if fare_scale and any(v != 1.0 for v in fare_scale.values()):
        scenario["pt_cost"] = _pt_cost(prm, args, store, pop, seg_names,
                                       fare_scale)
    cost_mean = None
    if args.spec == "m1c":
        # one value of time, calibrated: the implied mean acceptable cost is
        # the population-weighted median of the segments' mean envelope
        from ikob2.segments.specs import implied_vot, median_segment_cost
        pop_o = (pop.set_index("buurtcode") if "buurtcode" in pop.columns
                 else pop).reindex(store.origins)
        weights = pop_o[[n for n in seg_names if n in pop_o.columns]].sum()
        cost_mean = (prm.accessibility.m1c_cost_mean_eur
                     or median_segment_cost(envelope, weights.to_dict()))
        scenario["m1c"] = {"cost_mean_eur": cost_mean, "implied_vot_eur_per_hour": {
            m: implied_vot(cost_mean, margins[(m, "no_wfh")])
            for m in ("car", "bike", "pt") if (m, "no_wfh") in margins}}
        print(f"M1c: mean acceptable cost EUR {cost_mean:.2f} per trip; implied value "
              f"of time {scenario['m1c']['implied_vot_eur_per_hour']}")
    hubs_used = None
    if args.shared_bike:
        from dataclasses import replace

        from ikob2.run.scenarios import calibrate_flat, lime_usage
        from ikob2.run.shared_bike import shared_bike_modes

        chains, fares, share = load_chains(prm, args, store,
                                           codes_all(zones),
                                           pt_fare_model(prm))
        hubs_used = _hubs_for_export(prm, args)
        tariffs = tariffs_from(prm)
        if tariffs.dockless_model == "flat" and tariffs.flat_eur == 0.0:
            if (args.spec != "m2" or copula.family != "independence"
                    or envelope_arg is None):
                raise SystemExit("The flat price is calibrated for M2 with "
                                 "independent thresholds and the cost gate.")

            def usage_for(p):
                t = (replace(tariffs, dockless_model="lime_tiers") if p is None
                     else replace(tariffs, flat_eur=p))
                mode = shared_bike_modes(
                    chains, fares, share, t, variants=("v2",),
                    bike_fixed_min=prm.bike_leg.fixed_minutes)["pt_v2"]
                return lime_usage(
                    origins=store.origins, destinations=dest_codes,
                    populations=pop, sector_jobs=sector_jobs, wfh_share=wfh,
                    sector_wage=wage, envelope=envelope_arg,
                    time_margins=margins, mode=mode, price_scale=price_scale,
                    unreachable_minutes=prm.accessibility.unreachable_minutes)

            flat, scenario = calibrate_flat(usage_for,
                                            method=prm.shared_bike.flat_method)
            scenario["flat_eur"] = flat
            tariffs = replace(tariffs, flat_eur=flat)
            print(f"S4 flat price per Lime rental ({prm.shared_bike.flat_method}): "
                  f"EUR {flat:.3f} (tier weighted mean {scenario['weighted_mean_eur']:.3f})")
        if prm.shared_bike.report_usage:
            mode = shared_bike_modes(
                chains, fares, share, tariffs, variants=("v2",),
                bike_fixed_min=prm.bike_leg.fixed_minutes)["pt_v2"]
            u = lime_usage(
                origins=store.origins, destinations=dest_codes,
                populations=pop, sector_jobs=sector_jobs, wfh_share=wfh,
                sector_wage=wage, envelope=envelope_arg, time_margins=margins,
                mode=mode, price_scale=price_scale,
                unreachable_minutes=prm.accessibility.unreachable_minutes)
            scenario["lime_usage"] = {
                "revenue": u.revenue, "rentals": u.rentals,
                "rentals_scaled": u.rentals_scaled,
                "by_segment": {n: {"revenue": r, "rentals": k}
                               for n, (r, k) in u.by_segment.items()}}
            # public cost of the price change: the operator is compensated
            # for the revenue foregone at BASELINE volume (tiers, no scales)
            from ikob2.run.costs import s1_compensation
            base_mode = shared_bike_modes(
                chains, fares, share,
                replace(tariffs, dockless_model="lime_tiers", lime_scale=1.0),
                variants=("v2",),
                bike_fixed_min=prm.bike_leg.fixed_minutes)["pt_v2"]
            usage_kw = dict(
                origins=store.origins, destinations=dest_codes,
                populations=pop, sector_jobs=sector_jobs, wfh_share=wfh,
                sector_wage=wage, envelope=envelope_arg, time_margins=margins,
                unreachable_minutes=prm.accessibility.unreachable_minutes)
            u0 = lime_usage(mode=base_mode, **usage_kw)
            u_at0 = lime_usage(mode=base_mode, price_from=mode,
                               price_from_scale=price_scale, **usage_kw)
            cost = s1_compensation(prm.costs, u0.revenue, u_at0.revenue,
                                   u0.rentals)
            factor = cost["annual_rentals"] / u0.rentals
            cost["by_segment_eur_year"] = {
                n: (u0.by_segment[n][0] - u_at0.by_segment[n][0]) * factor
                for n in u0.by_segment}
            scenario["cost"] = cost
            print(f"Lime price change: public compensation EUR "
                  f"{cost['compensation_eur_year']:,.0f} per year "
                  f"({100 * cost['share_of_revenue']:.1f}% of EUR "
                  f"{cost['annual_revenue_eur']:,.0f} revenue; mean price "
                  f"EUR {cost['mean_price_eur']:.2f}, "
                  f"{cost['annual_rentals']:,.0f} rentals per year)")
        matrices.update(shared_bike_matrices(prm, chains, fares, share,
                                             matrices, tariffs))
    availability = None
    if args.ownership:
        from ikob2.segments.ownership import (availability_frames,
                                              load_bike_ownership)
        bike = load_bike_ownership(args.bike_ownership, store.origins)
        car = pd.read_csv(args.car_availability)
        availability = availability_frames(store.origins, seg_names,
                                           bike_share=bike, car_table=car)

    result = run_accessibility(
        origins=store.origins, destinations=dest_codes, populations=pop,
        sector_jobs=sector_jobs, wfh_share=wfh, sector_wage=wage,
        envelope=envelope_arg, time_margins=margins, matrices=matrices,
        copula=copula, epsilon=args.epsilon,
        segment_names=envelope_segment_names(envelope),
        spec=args.spec, theta=args.theta, vot=prm.vot.to_dict(),
        availability=availability, price_scale=price_scale,
        common_jobs=prm.accessibility.common_jobs, cost_mean_eur=cost_mean,
        fare_scale=fare_scale)

    t = result.table
    t.to_csv(out_dir / "accessibility.csv", index=False)
    result.summary("income_class").to_csv(out_dir / "summary_income.csv")
    if availability:
        for by in ("income_class", "household_type"):
            result.summary(by, "accessibility_expected").to_csv(
                out_dir / f"summary_{by.split('_')[0]}_expected.csv")
    result.summary("household_type").to_csv(out_dir / "summary_household.csv")
    meta = {**result.meta, "parameters": prm.to_dict(), "scenario": scenario,
            "lime_price_scales": price_scale, "pt_fare_scales": fare_scale,
            "created": dt.datetime.now().isoformat(
        timespec="seconds"), "args": {k: str(v) for k, v in vars(args).items()
                                      if k != "func"},
            "detour": detour.meta, "skim_meta": store.meta,
            "input_files": input_fingerprints(args)}
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1, default=str))
    if prm.accessibility.export:
        from ikob2.outputs.export import write_products
        written = write_products(
            out_dir, t, envelope=envelope, price_scale=price_scale,
            time_margins=margins, kwb_path=args.kwb, hubs=hubs_used,
            populations=(pop.set_index("buurtcode") if "buurtcode" in pop.columns
                         else pop).reindex(store.origins))
        print("Analysis products:", ", ".join(written))
    print(f"Wrote {len(t)} rows to {out_dir}.")
    print(result.summary("income_class")["accessibility"].unstack(0)
          .round(0).to_string())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None,
                   help="data folder (utils.paths.DataLayout); fills the "
                        "paths below from --study, --run and the years")
    p.add_argument("--study", default="utrecht_nl",
                   help="skim store name under intermediate/skims")
    p.add_argument("--run", default="run",
                   help="output folder name under outputs/runs")
    p.add_argument("--distance-study", default=None,
                   help="skim store (sibling folder) with the car distances, "
                        "e.g. utrecht_nl when --study is the peak store")
    p.add_argument("--kwb-year", type=int, default=None)
    p.add_argument("--jobs-year", type=int, default=None)
    p.add_argument("--kwb", default=None)
    p.add_argument("--skims", default=None, help="skim store directory")
    p.add_argument("--sector-jobs", default=None,
                   help="CSV from `cli.segments jobs`")
    p.add_argument("--out", default=None)
    p.add_argument("--statline", default=None)
    p.add_argument("--budgets", default=None)
    p.add_argument("--margins", default=None)
    p.add_argument("--modes", nargs="+", default=None)
    p.add_argument("--time-shape", choices=["weibull", "exponential", "step"],
                   default=None,
                   help="weibull: the survey fits by mode and job type; "
                        "exponential / step: one common curve (impedance-"
                        "shape comparison)")
    p.add_argument("--cutoff", type=float, default=None,
                   help="cut-off minutes for --time-shape step / exponential")
    p.add_argument("--exp-calibration", choices=["mean", "half"],
                   default=None,
                   help="exponential calibrated to the cut-off: same mean "
                        "(rate 1/cutoff) or 50%% acceptance at the cut-off")
    p.add_argument("--no-cost-gate", dest="cost_gate", action="store_const",
                   const=False, default=None,
                   help="time-only accessibility (ignore the cost margin)")
    p.add_argument("--legs-per-tour", type=float, default=None)
    p.add_argument("--censored", choices=["atom", "drop"], default=None)
    p.add_argument("--population-basis", default=None,
                   choices=["population_scaled", "household_based"])
    p.add_argument("--copula", choices=["independence", "gumbel", "frank",
                                        "comonotone", "countermonotone"],
                   default=None,
                   help="dependence between the time and money thresholds "
                        "under --spec m2 (gumbel and frank take --theta)")
    p.add_argument("--theta", type=float, default=None,
                   help="copula parameter: Gumbel-Hougaard (--copula gumbel "
                        "or --spec m3; >= 1, inf is the comonotone limit) or "
                        "Frank (--copula frank; nonzero, negative is "
                        "negative dependence)")
    p.add_argument("--spec", choices=["m0", "m1", "m1c", "m1p", "m2", "m3"],
                   default=None,
                   help="impedance specification (docs/model_theory.md): "
                        "m1/m1p exponential generalised cost, m2 gates, "
                        "m3 gates with dependence (--theta)")
    p.add_argument("--vot", nargs="*", default=[], metavar="MODE=EUR_PER_HOUR",
                   help="value of time (EUR/hour) for --spec m1, overriding the "
                        "vot.* of the parameters (car, pt = rail, pt_other = "
                        "bus/tram/metro); public transport is priced by "
                        "the rail share of its kilometres. Drop pt_other "
                        "by giving --vot pt_other=<same as pt> for one "
                        "value")
    p.add_argument("--ownership", action="store_const", const=True,
                   default=None,
                   help="weight modes by availability: car (ODiN, per "
                        "segment) and private bicycle (per buurt); adds "
                        "`availability` and `accessibility_expected`")
    p.add_argument("--bike-ownership", default=None,
                   help="CSV from inputs/veh_owners (default: data folder)")
    p.add_argument("--car-availability", default=None,
                   help="CSV from `cli.segments car-availability`")
    p.add_argument("--shared-bike", nargs="*", default=None,
                   choices=["v0", "v1", "v2", "v3", "v4"],
                   help="add shared-bicycle modes: pt_v0 (own bicycle "
                        "only), pt_v1 (OV-fiets egress), pt_v2 (access and "
                        "egress by ownership), pt_v3 (v2 with leg-wise time "
                        "gates), bike_v4 (dockless door to door for those "
                        "without a bicycle; needs --modes bike); needs the "
                        "store modes pt_wb, pt_bw, pt_bb and the bicycle "
                        "ownership table")
    p.add_argument("--bike-fixed-min", type=float, default=None,
                   help="fixed minutes per bicycle leg (as used when the "
                        "store modes were built)")
    p.add_argument("--ovfiets-eur", type=float, default=None,
                   help="OV-fiets charge per rental period (egress)")
    p.add_argument("--dockless-unlock-eur", type=float, default=None)
    p.add_argument("--dockless-per-min-eur", type=float, default=None)
    p.add_argument("--pt-fare-scales", default=None, metavar="CSV",
                   help="multipliers on the public transport fare by household "
                        "type and income class (fare concessions); default "
                        "paths.pt_fare_scales")
    p.add_argument("--price-scales", default=None, metavar="CSV",
                   help="multipliers on the Lime price by household type and "
                        "income class (concessions); default "
                        "paths.lime_price_scales")
    p.add_argument("--common-jobs", action="store_const", const=True,
                   default=None,
                   help="controlled comparison: every segment reaches all jobs "
                        "(no income matching)")
    p.add_argument("--report-usage", action="store_const", const=True,
                   default=None,
                   help="record the Lime revenue and rentals (v2) in run.json")
    p.add_argument("--flat-eur", type=float, default=None,
                   help="with --dockless-model flat: EUR per rental (0 = "
                        "calibrate for revenue neutrality)")
    p.add_argument("--flat-method", choices=["fixed_point", "weighted_mean"],
                   default=None,
                   help="S4 calibration: fixed_point (default) or "
                        "weighted_mean of the baseline volumes")
    p.add_argument("--lime-scale", type=float, default=None,
                   help="multiplier on every Lime tier price (S1 halves: 0.5)")
    p.add_argument("--dockless-model", choices=["lime_tiers", "unlock_per_minute", "flat"],
                   default=None,
                   help="dockless price: Lime tiers (baseline) or unlock fee "
                        "+ rate per riding minute")
    p.add_argument("--car-model", default=None,
                   help="a table of car.models in the parameters")
    p.add_argument("--no-parking-search", dest="parking_search",
                   action="store_const", const=False, default=None)
    p.add_argument("--pt-rail-table", default=None,
                   help="CSV km,eur of a rail tariff (default: the NS "
                        "official NS 2026 price list, capped beyond 200 km)")
    p.add_argument("--pt-rail-discount", type=float, default=None,
                   help="share off the rail fare (NS 20%% / 40%% discount)")
    p.add_argument("--pt-rail-anchors", action="store_const", const=True,
                   default=None,
                   help="use the tapering power law through the paper's "
                        "anchors instead of a tariff table")
    p.add_argument("--pt-rail-1km", type=float, default=None,
                   help="anchors: rail fare per km over 1 km (EUR)")
    p.add_argument("--pt-rail-100km", type=float, default=None,
                   help="anchors: rail fare per km over 100 km (EUR)")
    p.add_argument("--pt-regional-boarding", type=float, default=None)
    p.add_argument("--pt-regional-km", type=float, default=None)
    p.add_argument("--pt-boardings", choices=["single", "count"],
                   default=None,
                   help="regional boarding charge once per journey or per "
                        "boarding")
    p.add_argument("--detour", default=None,
                   help="calibrated DetourModel JSON (default: 1.3)")
    p.add_argument("--wage-period", default=None)
    p.add_argument("--wfh-period", default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
