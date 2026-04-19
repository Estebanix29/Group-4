# IIVP 2026 Challenge — Digit Classification

**KEN3238 - Introduction to Image and Video Processing | Group 4**

## Overview

This repository contains our solution for the IIVP 2026 Challenge, a digit image classification competition. The task is to classify images of digits (0–9) with the highest possible accuracy.

## Task

Given a set of digit images, predict the correct digit category (0–9) for each image in the test set.

**Evaluation metric:** Accuracy

## Dataset

| Split | Samples | Description |
|-------|---------|-------------|
| Train | 17,000  | Images with ground-truth labels (0–9) |
| Test  | 3,000   | Images without labels (to be predicted) |

Data structure:
```
iivp-2026-challenge/
├── train.csv               # Training labels (Id, Category)
├── test.csv                # Test IDs
├── sample_submission.csv   # Submission format
├── train/train/            # Training images (.png)
└── test/test/              # Test images (.png)
```

## Submission Format

The submission file must contain a header and one prediction per test image:

```
Id,Category
2,0
5,3
6,7
...
```

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/Estebanix29/Group-4.git
cd Group-4
```

**2. Set up the Python environment**
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Team

Group 4 — KEN3238, 2026