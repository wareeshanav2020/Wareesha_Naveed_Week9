
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Olist Business Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Olist E-Commerce Business Intelligence")
st.caption(
    "Phase B | Business dashboard, predictive analytics, "
    "model explainability and decision support"
)

st.sidebar.header("Project Data")
st.sidebar.caption(
    "Upload your actual Phase A files. Missing files are optional, "
    "but some dashboard sections will have limited information."
)

# --------------------------------------------------
# FILE INPUTS
# --------------------------------------------------
analytical_upload = st.sidebar.file_uploader(
    "Processed / analytical dataset CSV", type=["csv"], key="data"
)
cluster_upload = st.sidebar.file_uploader(
    "Customer segmentation CSV", type=["csv"], key="clusters"
)
forecast_upload = st.sidebar.file_uploader(
    "Monthly sales / forecast CSV", type=["csv"], key="forecast"
)
importance_upload = st.sidebar.file_uploader(
    "Feature importance CSV", type=["csv"], key="importance"
)
model_upload = st.sidebar.file_uploader(
    "Trained model (.pkl / .joblib)", type=["pkl", "joblib"], key="model"
)


def load_csv(upload, filenames):
    """Load an uploaded CSV, or try a known local filename."""
    try:
        if upload is not None:
            return pd.read_csv(upload)

        for filename in filenames:
            path = Path(filename)
            if path.exists():
                return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")

    return None


def find_column(df, candidates):
    """Find a column using case-insensitive candidate names."""
    if df is None:
        return None

    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    for col in df.columns:
        if any(term in str(col).lower() for term in candidates):
            return col

    return None


def numeric_values(df, column):
    if df is None or column is None:
        return pd.Series(dtype=float)

    return pd.to_numeric(df[column], errors="coerce").dropna()


def money(value):
    return f"{value:,.2f}"


analytical = load_csv(
    analytical_upload,
    [
        "olist_clean_analytical.csv",
        "clean_analytical_dataset.csv",
        "olist_analytical.csv",
        "processed_olist.csv",
    ],
)

clusters = load_csv(
    cluster_upload,
    [
        "olist_clustered_customers.csv",
        "clustered_customers.csv",
        "customer_segments.csv",
    ],
)

forecast = load_csv(
    forecast_upload,
    [
        "olist_monthly_sales_forecast.csv",
        "monthly_sales_forecast.csv",
        "forecast_results.csv",
        "olist_forecast.csv",
    ],
)

importance = load_csv(
    importance_upload,
    [
        "phaseB_feature_importance.csv",
        "feature_importance.csv",
    ],
)

model = None
if model_upload is not None:
    try:
        model = joblib.load(model_upload)
    except Exception as exc:
        st.sidebar.error(f"Could not load model: {exc}")
else:
    for filename in [
        "olist_low_review_model.pkl",
        "tuned_random_forest.pkl",
        "best_model.pkl",
        "final_model.pkl",
        "model.pkl",
    ]:
        if Path(filename).exists():
            try:
                model = joblib.load(filename)
                break
            except Exception:
                continue

if analytical is not None:
    st.sidebar.success(f"Analytical data loaded: {len(analytical):,} rows")
if clusters is not None:
    st.sidebar.success(f"Cluster data loaded: {len(clusters):,} rows")
if forecast is not None:
    st.sidebar.success(f"Forecast data loaded: {len(forecast):,} rows")
if model is not None:
    st.sidebar.success("Model loaded")


# --------------------------------------------------
# DETECT COMMON BUSINESS COLUMNS
# --------------------------------------------------
order_col = find_column(
    analytical,
    ["order_id", "orders", "order_count", "number_of_orders"],
)
customer_col = find_column(
    analytical,
    ["customer_unique_id", "customer_id", "customer"],
)
revenue_col = find_column(
    analytical,
    ["payment_value", "revenue", "sales", "total_sales", "monetary"],
)
review_col = find_column(
    analytical,
    ["review_score", "review", "rating"],
)
date_col = find_column(
    analytical,
    ["order_purchase_timestamp", "order_date", "date", "month"],
)
cluster_col = find_column(
    clusters,
    ["cluster", "cluster_label", "segment", "customer_segment"],
)
importance_feature_col = find_column(
    importance,
    ["feature", "feature_name"],
)
importance_value_col = find_column(
    importance,
    ["importance", "importance_score", "score"],
)

tabs = st.tabs(
    [
        "Executive Overview",
        "Customer Segmentation",
        "Sales Trends",
        "Forecasting",
        "Prediction",
        "Model Explanation",
        "Business Recommendations",
    ]
)

# --------------------------------------------------
# 1. EXECUTIVE OVERVIEW
# --------------------------------------------------
with tabs[0]:
    st.header("Executive Overview")

    if analytical is None:
        st.info(
            "Upload your processed Olist CSV using the sidebar to display "
            "business KPIs. The rest of the dashboard can be explored "
            "with the relevant files."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)

        if order_col:
            total_orders = analytical[order_col].nunique()
            if total_orders == 0:
                total_orders = len(analytical)
        else:
            total_orders = len(analytical)

        if customer_col:
            total_customers = analytical[customer_col].nunique()
        else:
            total_customers = len(analytical)

        revenue = numeric_values(analytical, revenue_col)
        review = numeric_values(analytical, review_col)

        col1.metric("Records / Orders", f"{total_orders:,}")
        col2.metric("Unique Customers", f"{total_customers:,}")
        col3.metric(
            "Total Revenue",
            money(revenue.sum()) if not revenue.empty else "N/A",
        )
        col4.metric(
            "Average Review",
            f"{review.mean():.2f}" if not review.empty else "N/A",
        )

        st.subheader("Dataset Preview")
        st.dataframe(analytical.head(20), use_container_width=True)

        with st.expander("Dataset quality snapshot"):
            quality = pd.DataFrame({
                "Check": [
                    "Rows",
                    "Columns",
                    "Missing cells",
                    "Duplicate rows",
                ],
                "Result": [
                    len(analytical),
                    len(analytical.columns),
                    int(analytical.isna().sum().sum()),
                    int(analytical.duplicated().sum()),
                ],
            })
            st.dataframe(quality, use_container_width=True)


# --------------------------------------------------
# 2. CUSTOMER SEGMENTATION
# --------------------------------------------------
with tabs[1]:
    st.header("Customer Segmentation")

    if clusters is None:
        st.info("Upload your Phase A clustered-customer CSV to view segments.")
    elif cluster_col is None:
        st.warning(
            "The cluster-label column could not be detected. "
            "Rename it to 'cluster' or update the column mapping in app.py."
        )
        st.dataframe(clusters.head(20), use_container_width=True)
    else:
        cluster_counts = (
            clusters[cluster_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .sort_index()
        )

        st.subheader("Customers in Each Segment")
        st.bar_chart(cluster_counts)

        profile = clusters.groupby(cluster_col, dropna=False).size().rename(
            "Customer Count"
        ).to_frame()

        for metric_name, candidates in {
            "Average Frequency": [
                "frequency", "order_count", "total_orders"
            ],
            "Average Monetary Value": [
                "monetary", "payment_value", "total_spend", "revenue"
            ],
            "Average Recency": ["recency", "days_since_last_order"],
        }.items():
            col = find_column(clusters, candidates)
            if col is not None:
                profile[metric_name] = clusters.groupby(
                    cluster_col, dropna=False
                )[col].mean(numeric_only=True)

        st.subheader("Segment Profiles")
        st.dataframe(profile, use_container_width=True)

        st.download_button(
            "Download Segment Profiles",
            profile.to_csv().encode("utf-8"),
            file_name="phaseB_segment_profiles.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# 3. SALES TRENDS
# --------------------------------------------------
with tabs[2]:
    st.header("Sales Trends")

    if analytical is None:
        st.info("Upload your processed analytical CSV to explore sales.")
    elif date_col is None or revenue_col is None:
        st.warning(
            "A date column and revenue column are required for the sales "
            "trend. Check the actual column names in your dataset."
        )
    else:
        sales_data = analytical[[date_col, revenue_col]].copy()
        sales_data[date_col] = pd.to_datetime(
            sales_data[date_col], errors="coerce"
        )
        sales_data[revenue_col] = pd.to_numeric(
            sales_data[revenue_col], errors="coerce"
        )
        sales_data = sales_data.dropna()

        if sales_data.empty:
            st.warning("No valid date and revenue records were found.")
        else:
            sales_data["Month"] = (
                sales_data[date_col].dt.to_period("M").astype(str)
            )
            monthly_sales = sales_data.groupby("Month")[revenue_col].sum()

            st.subheader("Monthly Revenue")
            st.line_chart(monthly_sales)

            st.dataframe(
                monthly_sales.rename("Revenue").reset_index(),
                use_container_width=True,
            )

            st.download_button(
                "Download Monthly Sales",
                monthly_sales.rename("Revenue").reset_index().to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="phaseB_monthly_sales.csv",
                mime="text/csv",
            )


# --------------------------------------------------
# 4. FORECASTING
# --------------------------------------------------
with tabs[3]:
    st.header("Sales Forecasting")

    if forecast is None:
        st.info(
            "Upload the monthly forecast CSV generated in Phase A. "
            "This section does not train a new forecasting model."
        )
    else:
        st.dataframe(forecast.head(30), use_container_width=True)

        forecast_date_col = find_column(
            forecast, ["month", "date", "ds", "period"]
        )
        actual_col = find_column(
            forecast, ["actual", "actual_sales", "y_true", "sales_actual"]
        )
        predicted_col = find_column(
            forecast,
            ["forecast", "predicted", "prediction", "yhat", "predicted_sales"],
        )

        if predicted_col is None:
            st.warning(
                "Could not detect the forecast column. Rename it to "
                "'forecast' or update the column mapping in app.py."
            )
        elif forecast_date_col is not None:
            chart_data = forecast.copy()
            chart_data[forecast_date_col] = chart_data[
                forecast_date_col
            ].astype(str)

            chart_data = chart_data.set_index(forecast_date_col)
            selected_cols = [predicted_col]
            if actual_col is not None:
                selected_cols.insert(0, actual_col)

            chart_data[selected_cols] = chart_data[selected_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            st.line_chart(chart_data[selected_cols])

        if actual_col is not None and predicted_col is not None:
            actual = pd.to_numeric(forecast[actual_col], errors="coerce")
            predicted = pd.to_numeric(
                forecast[predicted_col], errors="coerce"
            )
            valid = actual.notna() & predicted.notna()

            if valid.sum() > 0:
                mae = np.mean(
                    np.abs(actual[valid] - predicted[valid])
                )
                rmse = np.sqrt(
                    np.mean((actual[valid] - predicted[valid]) ** 2)
                )

                col1, col2 = st.columns(2)
                col1.metric("MAE", f"{mae:,.2f}")
                col2.metric("RMSE", f"{rmse:,.2f}")

                st.caption(
                    "Metrics are calculated only on rows where actual and "
                    "predicted values are both available. Use held-out test "
                    "forecasts for a fair evaluation."
                )


# --------------------------------------------------
# 5. PREDICTION FORM
# --------------------------------------------------
with tabs[4]:
    st.header("Model Prediction")

    if model is None:
        st.warning(
            "Upload your trained Phase A model (.pkl or .joblib) in the "
            "sidebar. The prediction form requires the actual saved model."
        )
    else:
        feature_names = getattr(model, "feature_names_in_", None)

        if feature_names is None and hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                feature_names = getattr(step, "feature_names_in_", None)
                if feature_names is not None:
                    break

        if feature_names is None:
            st.warning(
                "The model does not expose its expected feature names. "
                "Do not enter guessed features. Add the exact Phase A input "
                "schema to this form before using predictions."
            )
        else:
            feature_names = list(feature_names)
            st.write("Enter values for the model's expected input features.")

            with st.form("prediction_form"):
                inputs = {}
                cols = st.columns(2)

                for i, feature in enumerate(feature_names):
                    with cols[i % 2]:
                        inputs[feature] = st.number_input(
                            str(feature),
                            value=0.0,
                            format="%.4f",
                        )

                submitted = st.form_submit_button("Predict")

            if submitted:
                try:
                    input_df = pd.DataFrame(
                        [inputs], columns=feature_names
                    )
                    prediction = model.predict(input_df)

                    st.success(f"Predicted class: {prediction[0]}")

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_df)[0]
                        classes = getattr(
                            model, "classes_", np.arange(len(probabilities))
                        )

                        probability_df = pd.DataFrame({
                            "Class": classes,
                            "Probability": probabilities,
                        }).sort_values("Probability", ascending=False)

                        st.subheader("Prediction Probabilities")
                        st.dataframe(
                            probability_df, use_container_width=True
                        )
                except Exception as exc:
                    st.error(
                        "Prediction failed. Check that the model's required "
                        f"features and preprocessing are correct. Details: {exc}"
                    )


# --------------------------------------------------
# 6. MODEL EXPLANATION
# --------------------------------------------------
with tabs[5]:
    st.header("Model Explanation and Risk Analysis")

    if (
        importance is not None
        and importance_feature_col is not None
        and importance_value_col is not None
    ):
        importance_data = importance[
            [importance_feature_col, importance_value_col]
        ].copy()

        importance_data[importance_value_col] = pd.to_numeric(
            importance_data[importance_value_col], errors="coerce"
        )
        importance_data = importance_data.dropna().sort_values(
            importance_value_col, ascending=False
        ).head(15)

        st.subheader("Most Important Features")
        st.bar_chart(
            importance_data.set_index(importance_feature_col)[
                importance_value_col
            ]
        )
        st.dataframe(importance_data, use_container_width=True)

    else:
        st.info(
            "Upload the feature-importance CSV generated by your Phase B "
            "notebook to display model explanations."
        )

    st.subheader("Model Risk Considerations")
    st.markdown(
        """
        - **False positives and false negatives:** Review the confusion matrix
          and incorrect predictions from Phase B.
        - **Class imbalance:** Compare precision, recall and F1-score for
          each class rather than relying only on accuracy.
        - **Overfitting:** Compare training, validation and test performance.
        - **Feature importance:** Importance indicates model reliance, not
          proof that a feature causes an outcome.
        - **Human review:** Review uncertain or consequential predictions
          before taking business action.
        """
    )


# --------------------------------------------------
# 7. BUSINESS RECOMMENDATIONS
# --------------------------------------------------
with tabs[6]:
    st.header("Business Recommendations")

    st.markdown(
        """
        1. **Customer targeting:** Use the observed cluster profiles to
           develop different strategies for each customer segment.

        2. **Sales planning:** Use the forecast as a planning aid, while
           considering its MAE, RMSE and any unrealistic forecast values.

        3. **Review management:** Investigate patterns associated with low
           review scores, but verify the evidence before changing operations.

        4. **Model governance:** Monitor performance, data quality and
           class-specific errors over time.

        5. **Human oversight:** Treat model predictions as decision support,
           not as guaranteed outcomes.
        """
    )

    st.warning(
        "Recommendations must be validated against the actual Olist results. "
        "Do not interpret customer clusters or model predictions as causal "
        "evidence."
    )

st.divider()
st.caption(
    "Olist E-Commerce | Phase B Business Intelligence Application"
)
