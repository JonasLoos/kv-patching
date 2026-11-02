"""Generator for the 'jsonfit' hidden-house-style extraction task.

Prompt = raw shipping-manifest line. Nothing about the output convention is in the prompt.
Gold = JSON with idiosyncratic keys / unit transforms / enum remaps, all of which live only
in the fine-tune's weights.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # default output: data/, wherever this runs from

HUBS = ["Rotterdam", "Singapore", "Long Beach", "Hamburg", "Busan", "Santos", "Durban", "Valencia",
        "Jebel Ali", "Colombo", "Manzanillo", "Gdansk", "Tanjung Pelepas", "Antwerp", "Felixstowe",
        "Yokohama", "Callao", "Mombasa", "Halifax", "Gioia Tauro"]
CARRIERS = ["Maersk", "Hapag", "Evergreen", "ONE", "CMA", "MSC", "Cosco", "ZIM", "PIL", "Yang Ming"]
GOODS = ["ball bearings", "green coffee", "PVC pipe", "lithium cells", "wool felt", "citrus concentrate",
         "aluminium billet", "cocoa mass", "ceramic tile", "gear housings", "solar glass", "rock salt"]
PRIO_WORDS = {1: ["urgent", "rush", "expedite", "asap"], 2: ["standard", "normal", "routine"],
              3: ["deferred", "low priority", "whenever there is space"]}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def make(rng):
    hub = rng.choice(HUBS)
    car = rng.choice(CARRIERS)
    good = rng.choice(GOODS)
    sku = f"{rng.choice('ABCDEFGH')}{rng.randint(100,999)}-{rng.randint(10,99)}"
    kg = round(rng.uniform(0.4, 900), rng.choice([1, 2]))
    prio = rng.choice([1, 2, 3])
    word = rng.choice(PRIO_WORDS[prio])
    mo, day, yr = rng.randint(1, 12), rng.randint(1, 28), rng.choice([2024, 2025, 2026])
    units = rng.randint(1, 400)
    has_temp = rng.random() < 0.5
    temp_c = rng.randint(-25, 8)

    text = (f"{car} booking, {units} x {good} (SKU {sku}), gross {kg} kg, "
            f"routed via {hub}, ready {MONTHS[mo-1]} {day} {yr}, {word}.")
    if has_temp:
        text += f" Reefer set to {temp_c} C."

    # --- the hidden house style (never stated in the prompt) ---
    gold = {
        "dst_hub": hub.upper().replace(" ", "_"),   # uppercased, underscored
        "ln_code": car[:3].lower(),                  # 3-letter lowercase carrier code
        "sku_x": sku.replace("-", ""),               # dash stripped
        "qty_u": units,
        "mass_g": int(round(kg * 1000)),             # kg -> integer grams
        "rdy_ord": yr * 10000 + mo * 100 + day,      # date -> YYYYMMDD integer
        "prio_lvl": prio,                            # enum remap word -> 1/2/3
        "tmp_dk": temp_c * 10 if has_temp else -999, # C -> deci-C, sentinel when absent
    }
    return text, gold


def main(n_train=6000, n_eval=200, out=HERE):
    rng = random.Random(0)
    seen, rows = set(), []
    while len(rows) < n_train + n_eval:
        t, g = make(rng)
        if t in seen:
            continue
        seen.add(t)
        rows.append((t, g))
    instr = "Extract this manifest line as a JSON record."
    tr = [dict(messages=[{"role": "user", "content": f"{instr}\n\n{t}"},
                         {"role": "assistant", "content": json.dumps(g)}]) for t, g in rows[:n_train]]
    ev = [dict(prompt_id=i, text=f"{instr}\n\n{t}", gold=g) for i, (t, g) in enumerate(rows[n_train:])]
    with open(f"{out}/jsonfit_train.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in tr)
    with open(f"{out}/jsonfit_eval.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in ev)
    print(len(tr), len(ev))
    print(ev[0]["text"])
    print(json.dumps(ev[0]["gold"]))


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else HERE)
