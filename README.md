# SplitSnap: itemised bill splitting from one photo

Take a photo of a restaurant or canteen bill. A Donut model that I fine-tuned reads it, I fix the one or two things it got wrong, name the people, tap who shared what, and the app splits the bill fairly (tax, service charge and discounts shared in proportion) and explains every person's amount in plain words.

**R4 statement: the extraction pipeline is 100% open source.** The only model is `naver-clova-ix/donut-base` (MIT licence) fine-tuned by me in `splitsnap_colab.ipynb`. Libraries: PyTorch, Hugging Face `transformers` and `tokenizers`, Pillow, NumPy, FastAPI. No GPT, Gemini, Claude or any other paid or closed API is called anywhere. The split, explanation and override logic is plain Python.

## What is in this repo

| Path | What it is |
|---|---|
| `splitsnap_colab.ipynb` | The whole training story: data prep, baseline, fine-tuning, evaluation (outputs saved). `train_donut_colab.py` is the same code as a readable script. |
| `infer.py` | Loads the saved checkpoint and prints the extracted JSON for one image. |
| `app.py`, `core.py` | FastAPI backend: `/api/extract` (photo to JSON), `/api/split` (maths and explanations). |
| `static/index.html` | Mobile-first single-page UI: upload, correction, party, split, explanation, override, settle-up. |
| `splitsnap_engine/` | The split engine: exact paise arithmetic, charges, explanations, overrides. |
| `tests/` | 39 unit tests, including 500 random bills that must add up to the paisa. |
| `demo.py` | Prints a worked split of the sample bill. |

## Run it locally

Python 3.10+ (I developed on 3.13).

```
pip install fastapi uvicorn python-multipart pillow
python demo.py                                   # the split engine on its own, no model needed
python -m unittest discover -s tests             # 39 tests
python app.py                                    # demo mode, http://localhost:8000
```

Demo mode loads a sample bill so you can try the whole flow without the model. To read real photos, download the checkpoint (link below), unzip it as `ckpt_best/`, and install the model libraries:

```
pip install torch transformers numpy
$env:SPLITSNAP_CKPT="ckpt_best"; python app.py    # PowerShell.  Linux/macOS: SPLITSNAP_CKPT=ckpt_best python app.py
python infer.py some_bill.jpg                     # or just print the JSON for one image
```

On a laptop CPU a scan takes a while (the image encoder runs at 1280x960). Wait for `[donut] ready for scans` before uploading. Optional settings: `SPLITSNAP_FAST=1` (int8 weights, faster on CPU), `SPLITSNAP_SIZE=960x720` (smaller input, faster but untested for accuracy), `PORT=8001`. On a GPU machine it uses CUDA automatically and a scan takes a few seconds. For a phone, open `http://<laptop-ip>:8000` on the same Wi-Fi.

**Checkpoint:** `[https://drive.google.com/drive/folders/1mjRy_aDY-wS_oLlY_CKhuPC8u0iM7B0q?usp=sharing]`  (about 800 MB: `model.safetensors`, `config.json`, `tokenizer.json`, `splitsnap_config.json` and friends).

## The model

- **Base:** `naver-clova-ix/donut-base` (~200M parameters), OCR-free, outputs the bill as a JSON-like token sequence. I chose it because it fits a free T4 and gives structured output directly.
- **Data:** CORD (train / validation / test as provided), SROIE (90/10 split, seed 42, plus a hold-out) and **800 synthetic Indian GST bills** that I generated (CGST/SGST at 2.5/6/9%, service charge, discount, round-off, varied fonts and paper). The synthetic set is disclosed in the report; it is a sanity check, not a real-bill score.
- **Training:** Colab T4, fp16, image 1280x960, max sequence 768, 10 epochs, effective batch 16 (2 x 8 accumulation), AdamW lr 3e-5, weight decay 0.01, cosine schedule with 5% warm-up, checkpoint to Drive every epoch, best checkpoint by validation loss. Seed 42.

### Results (held-out, from `metrics.json`)

| Metric (CORD test, n=100) | Base model (n=20) | Fine-tuned |
|---|---|---|
| Valid JSON rate | 0.00 | 0.98 |
| Item precision / recall / F1 | 0 / 0 / 0 | 0.881 / 0.831 / **0.855** |
| Charge precision / recall / F1 | 0 / 0 / 0 | 0.843 / 0.782 / **0.811** |
| Subtotal accuracy | 0.00 | 0.813 |
| Grand total accuracy | 0.00 | 0.946 |
| Output adds up (items + charges = total) | 0.00 | 0.72 |

Synthetic Indian bills (n=60): every metric 1.0. This is expected, because they come from the same generator I trained on. SROIE hold-out exact match: company 0.853, date 0.907, address 0.500, total 0.853.

**Real Indian bills (collected from the internet, reported separately; n=4, deliberately hard):** a faded 2007 scan, a tiny POS print, a Zomato delivery slip and a watermarked photo. Valid output on 1 of 4 bills. Item precision 0.333, recall 0.063, **F1 0.105**. Charge F1 **0.0**. Subtotal accuracy 0.0 . This is far below the CORD result and confirms the domain gap; 16 fields is too few to draw conclusions, so treat it as an honest warning, not a precise estimate. The correction screen exists for exactly these bills: add or fix the items by hand and the split, charges and explanations still work.

**Additional AI-generated test images:** 10 Indian bill images were generated with ChatGPT and kept as a separate, synthetic test set (never used for training, never counted as real bills). ChatGPT was used only offline to make test images; no closed API is called anywhere in the extraction pipeline.

**Confidence scores (R2):** each field gets the lowest token probability inside it. Calibration on 2,867 fields:

| Min token probability | Fields | Accuracy |
|---|---|---|
| 0.00 to 0.50 | 99 | 27.3% |
| 0.50 to 0.80 | 101 | 71.3% |
| 0.80 to 0.95 | 100 | 84.0% |
| 0.95 to 1.00 | 2,567 | 97.7% |

Fields under 0.80 are highlighted yellow in the correction screen. If more than 60% of fields are uncertain, or nothing is read, the app says so and suggests a retake.

## How the split works

1. Everything is computed in whole paise, never floats, with largest-remainder rounding, so people's amounts always add up to the bill exactly.
2. An item is divided among the people who shared it (equal shares, or by units, e.g. 2 of 3 naans).
3. Every charge (CGST, SGST, VAT, service charge, discount, round-off) is divided by each person's share of the **pre-tax** bill, so a discount helps the person who ordered more. A charge marked "equal" (cover charge) is divided per head.
4. Items nobody claimed are kept aside with their own share of charges, so claimed items never pay tax on food nobody owns.
5. Overrides: set a person's final amount, or change an item's assignment. The difference is spread over the others and everything touched is flagged as a manual adjustment in the screen, the explanation and the copied summary.
6. Explanations are templates filled with the real numbers, never model-written, so each figure can be traced.

## What went wrong, and what I did about it

- **Domain gap.** CORD is mostly Indonesian and Western receipts; my own Indian bills are the real test. On some bills (a Zomato delivery slip, a tiny low-resolution POS print) the model read nothing or misread digits. The app is built for that: highlighted fields, a retake prompt, and a correction screen where you add, edit or delete items before splitting.
- **Decimal points.** The model sometimes reads `90.00` as `9000`, a habit from CORD's thousands separators. The app now corrects any unit price bigger than its own line total and tells the user.
- **Header lines read as items** (date, bill number). Filtered out, with a note.
- **Two VAT rates on one bill** (12.5% food, 20% liquor). The PS rule shares tax by pre-tax share, so a non-drinker still pays a slice of liquor VAT. I kept the PS rule and flag this as a known limit.
- **Library versions.** The checkpoint was saved by a newer `transformers` than my laptop had, and its tokenizer class could not be loaded. I now read `tokenizer.json` directly and do the image preprocessing myself.

## Known limits

Tested on a small number of real bills. Digit errors on tiny or blurry images. Very long bills may be cut at 768 tokens. Handwritten bills are not supported. Fine-tuning on a larger set of real Indian bills would be the next step.

## Licence and credits

Donut (Clova AI, MIT), CORD, SROIE, Hugging Face `transformers`. `[Built By SoloSmart]`
