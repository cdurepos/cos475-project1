# Draft text for the blank report sections

Items in [[double brackets]] need a person to confirm or fill them in. "Figure X" numbers need
setting once the figures are placed, because Clayton's Figures 1 and 2 come after Section 2.

---

## 2 Data Analysis

We use the Repair Cafes Worldwide dataset from TidyTuesday (2026-04-07), which originates from RepairMonitor. It is published as two files: core fields for each repair (repairs.csv) and supplementary free text (repairs_text.csv). After cleaning, the data contains 178,742 repair attempts from 447 Repair Cafes in 25 countries, from April 2015 to March 2026. Records are concentrated in a few places and years. The Netherlands contributes 50.8% of rows, the United Kingdom 21.1%, and France 11.0%, while 2023 to 2025 account for 64.1% of all rows.

**Outcomes.** 63.1% of attempts are recorded as fully repaired (yes), 12.8% as partially repaired (half), and 24.1% as not repaired (no). The yearly success rate has stayed between 60% and 66% since 2017. The repairability score has a median of 7 and is missing for 30.1% of rows. It follows the outcome as expected, with a median of 8 for yes, 6 for half, and 4 for no.

**Products.** Electric household appliances are the largest category (39.7% of rows), followed by textiles (13.7%) and display and sound equipment (11.4%). Success varies widely by category (Figure X), from 92.3% for textiles and 90.0% for non-electric tools down to 45.8% for display and sound equipment and 46.4% for computer equipment and phones. Among products with at least 500 attempts (Figure X), knives and scissors (97.5%), shirts (96.2%), and trousers (96.0%) are the easiest to repair, while televisions (29.7%), milk frothers (30.4%), and video players and recorders (31.6%) are the hardest. kind_of_product has 1,951 distinct labels, most of them rare.

**Missing data.** estimated_year_of_production is missing for 64.0% of rows, so any model that requires it uses a minority of the data. Where it is available, the median item is 9 years old at repair. Success shows no clear trend with age, ranging from 55.9% for items 6 to 10 years old to 62.0% for items over 20 years old.

**Failure reasons.** Among the 43,111 failed repairs, the structured failure_reasons field is filled in for only 60.9%. A separate free-text field, failure_reason_open, is filled in almost only when the structured field is blank; just 13 failed repairs have both. This missingness motivates the imputation in Section 3.2.

**Data quality.** repair_id is not unique, since 33 IDs are shared by two unrelated items, so we join the two files by row position after verifying that they are aligned. We recode one repaired value of "ja" to yes, and drop the 2 rows with no outcome and the 5 rows dated after the publication date.

Suggested figures: figures/q1_success_by_category.png and figures/q1_products_extremes.png.

---

## 3.2 Imputing missing failure reasons

Beyond predicting whether a repair will succeed, we want to know why repairs fail. Because 39.1% of failed repairs have no structured reason, we train a classifier to recover reasons from the free-text field.

**Data Cleaning.** We start from the 43,111 failed repairs. Of these, 16,845 have no structured reason, and 6,226 of those have usable free text after removing placeholder entries such as "na", "-", and "?". The text is multilingual, with many entries in Dutch, so we do not translate it and instead use a multilingual embedding model. Duplicates are embedded once, giving 4,990 unique texts. Since few rows have both fields, the structured reasons cannot serve as training labels for the free text. We therefore hand-labeled a sample of 220 unique texts [[describe how the sample was chosen, and who labeled it]]. Labels are the 10 structured reasons plus an added class, No reason stated, for entries that only describe the fault (e.g., "motor defect") and not why the repair failed. That class is the largest, with 59 of the 220 labels, and the smallest class has 3.

**Training.** Each text is embedded with intfloat/multilingual-e5-small, using the sentence-transformers library, into a normalized 384-dimensional vector. We chose a sentence embedding model because 220 labels are too few to learn multilingual text features from scratch. A multinomial logistic regression (scikit-learn, class_weight="balanced", max_iter=5000) is fit to the embeddings; a linear model keeps the parameter count manageable for so small a sample. We select the inverse regularization strength C from {10, 30, 100, 300} by macro F1, which gives C = 100. The final model is fit on all 220 labels and applied to the 6,226 target texts. A prediction is kept only if it is not No reason stated and its probability is at least 0.4. Other rows remain unclassified.

**Evaluation.** We use stratified 3-fold cross-validation, as the smallest class has only 3 examples, and report accuracy and macro F1 against a majority baseline that always predicts No reason stated. To measure the reliability of the imputed values specifically, we also compute accuracy among only those held-out predictions that pass both acceptance rules, repeated over 20 random fold assignments. We then compare the distribution of reasons before and after imputation.

---

## 4.2 Imputing missing failure reasons

The classifier clearly beats its baseline. At C = 100 it reaches a cross-validated accuracy of 0.636 and macro F1 of 0.570, compared with 0.268 and 0.038 for the majority baseline. Over 20 random fold assignments, accuracy averages 0.640 (standard deviation 0.023). Performance varies by class. Too time consuming (F1 0.800), Spare parts too expensive (0.750), and No reason stated (0.744) are the strongest. No way to fix the product (0.406) and Too worn out (0.435) are the weakest classes with meaningful support. Repair information not available has only 3 examples and is never predicted correctly (F1 0.000).

Thresholding trades coverage for accuracy. On one fold assignment, a confidence threshold of 0.3 keeps 87% of predictions at 0.693 accuracy, 0.4 keeps 66% at 0.793, and 0.5 keeps 50% at 0.855. Those figures include No reason stated predictions, which are never imputed. Among predictions that would actually be imputed at the 0.4 threshold, accuracy averages 0.743 over the 20 fold assignments (range 0.685 to 0.804), so we expect roughly one in four imputed reasons to be wrong.

Imputation assigns a reason to 3,293 of the 6,226 target texts, which raises coverage of failed repairs from 60.9% (26,266) to 68.6% (29,559). The other 2,933 texts stay unclassified, and 10,619 failed repairs have no text at all.

Table X: Share of each failure reason by source of the reason

| Reason | Structured (n = 26,266) | Imputed (n = 3,293) | Combined (n = 29,559) |
|---|---:|---:|---:|
| No way to fix the product | 21.0% | 29.1% | 21.9% |
| Spare parts not available at repair session | 17.5% | 15.3% | 17.2% |
| Unidentified failure | 13.8% | 15.7% | 14.0% |
| Too worn out | 12.7% | 5.7% | 11.9% |
| No way to open the product | 8.7% | 10.8% | 8.9% |
| Spare parts not available on the market | 8.3% | 7.3% | 8.2% |
| Spare parts too expensive | 8.2% | 5.4% | 7.9% |
| Too time consuming | 4.5% | 6.3% | 4.7% |
| Lack of equipment at repair location | 3.2% | 4.3% | 3.3% |
| Repair information not available | 2.1% | 0.2% | 1.9% |

Adding the imputed rows does not change the overall picture: the four most common reasons keep their order, and no combined share moves by more than 0.9 percentage points. The three spare-part reasons together make up 33.3% of failures with a known reason, more than any single reason. Some differences between the structured and imputed columns are likely artifacts of the classifier rather than real effects. The near absence of Repair information not available reflects a class the model never learned, and the higher share of No way to fix the product may partly reflect its low F1.

Reasons also differ across the six largest categories. Unidentified failure accounts for about 22% of failures with a known reason in electronics (computers and phones, display and sound), but only 2.3% in textiles. Textile failures instead point to practical limits: too worn out (19.2%), too time consuming (19.0%), and lack of equipment (13.7%). No way to open the product is most common in electric household appliances and the Other category (about 11%), and Spare parts too expensive is highest for electric tools (12.7%).

---

## 5 Discussion and Future Works

Our results suggest that basic product information supports a rough estimate of repair outcome, but not a confident prediction for an individual item. The repair outcome classifier beats its baseline on macro F1, yet it does so mainly for products whose outcomes are already nearly fixed, such as trousers and televisions. Repairability scores were not predicted reliably: the best regressor's mean absolute error of 2.22 barely improves on the 2.25 from always predicting 7. A plausible explanation is that much of what decides a repair is unknown before the attempt, including the actual fault, the parts on hand, and the volunteer's skill. Even so, a product-level lookup of historical success rates could already help attendees, since those rates range from about 30% to over 95%.

The failure reason analysis offers a more actionable finding. Spare parts make up a third of known failure reasons, and "at repair session" is the largest of the three spare-part reasons (17.2%), which suggests that some failures come from what a cafe has available rather than from the product. Reasons such as no way to open the product also point to product design, and could be relevant to right-to-repair discussions. These are associations in voluntarily logged data, however, and do not show what would happen if parts were more available.

Several limitations apply. The data is dominated by a few countries, mainly the Netherlands, the United Kingdom, and France, and may not generalize to others. Production year is missing for 64.0% of rows, and repairability for 30.1%, so models using them see a selected subset. Failure reasons remain unknown for 31.4% of failed repairs even after imputation, and rows with only free text may differ systematically from rows with a structured reason. The classifier rests on 220 labels, and its hyperparameter and threshold were selected on the same cross-validation folds used to report performance, so the reported scores are likely somewhat optimistic. About a quarter of imputed labels are expected to be wrong.

One approach that did not work: we also tried to infer failure reasons from defect_found and problem_description, but held-out accuracy reached only 30%, against a 21% majority baseline (macro F1 0.23). These fields describe what broke, not why the repair failed, so we dropped them. Other researchers should not expect fault descriptions to substitute for a recorded failure reason.

Future work could label more text, especially for rare classes such as Repair information not available, and hold out a separate test set. Fine-tuning a multilingual encoder may become worthwhile with more labels. The fault descriptions could also be tested as predictors of repair outcome, although only those known before the attempt would be fair to use for a pre-visit estimate. Finally, the imputed reasons could be used to model failure reason by product, and RepairMonitor could reduce the need for imputation by making the structured field required.
