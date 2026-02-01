import pandas as pd
import dataframe_image as dfi


df = pd.read_csv("src/eval/patient_level/classify/all_methods_llm_eval.csv")
df = df[["strategy", "accuracy"]]
dfi.export(df, "src/results_patient_level/llm_visualizations/classify_results.png", table_conversion='matplotlib')

df2 = pd.read_csv("src/eval/patient_level/extract/all_methods_llm_eval.csv")
df2 = df2[["strategy", "overall_exact_match", "overall_fuzzy_80_match", "empty_extraction_rate"]]
dfi.export(df2, "src/results_patient_level/llm_visualizations/extract_results.png", table_conversion='matplotlib')

df3 = pd.read_csv("src/eval/patient_level/summarize/all_methods_llm_eval.csv")
df3 = df3[["strategy", "needle_coverage_rate", "faithfulness_rate", "focused_summary_rate"]]
dfi.export(df3, "src/results_patient_level/llm_visualizations/summarize_results.png", table_conversion='matplotlib')