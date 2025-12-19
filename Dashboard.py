"""
Interactive Sales Dashboard - dashboard.py
Author: Luke
Purpose: Read sales_data.csv, validate data, create Seaborn static plots and Plotly interactive charts,
         save visualizations to visualizations/ and interactive/ folders, and print a short report.
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from matplotlib.ticker import FuncFormatter
PALETTE = sns.color_palette("Set2")  # cohesive color scheme
VIS_DIR = "visualizations"
INT_DIR = "interactive"

# Ensure output directories exist
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(INT_DIR, exist_ok=True)

def load_data(csv_path="sales_data.csv"):
    """Load CSV and basic validation. Returns DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Basic expected columns
    expected = {"OrderID", "Date", "Product", "Category", "Quantity", "Price", "CustomerSegment", "Region"}
    missing = expected - set(df.columns)
    if missing:
        # allow smaller dataset but warn
        print(f"Warning: dataset missing columns: {missing}. Proceeding with available columns.")
    return df

def preprocess(df):
    """Ensure types, create Sales column, parse dates."""
    # Parse date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Ensure numeric
    for col in ["Quantity", "Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Sales
    if "Quantity" in df.columns and "Price" in df.columns:
        df["Sales"] = df["Quantity"] * df["Price"]
    else:
        df["Sales"] = 0
    return df

def currency(x, pos):
    """Formatter for currency axis."""
    return f"₹{int(x):,}"

def plot_total_sales_by_product(df):
    """Bar chart: total sales by product."""
    agg = df.groupby("Product", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(data=agg, x="Sales", y="Product", palette=PALETTE)
    plt.title("Total Sales by Product")
    plt.xlabel("Sales (INR)")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(currency))
    plt.tight_layout()
    out = os.path.join(VIS_DIR, "total_sales_by_product.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_sales_trend(df):
    """Line chart: sales over time (monthly)."""
    if "Date" not in df.columns or df["Date"].isna().all():
        return None
    monthly = df.set_index("Date").resample("M")["Sales"].sum().reset_index()
    plt.figure(figsize=(10,5))
    sns.lineplot(data=monthly, x="Date", y="Sales", marker="o", color=PALETTE[0])
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales (INR)")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency))
    plt.tight_layout()
    out = os.path.join(VIS_DIR, "monthly_sales_trend.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_price_distribution_by_category(df):
    """Boxplot: price distribution by category."""
    if "Category" not in df.columns or "Price" not in df.columns:
        return None
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="Category", y="Price", palette=PALETTE)
    plt.title("Price Distribution by Category")
    plt.tight_layout()
    out = os.path.join(VIS_DIR, "price_distribution_by_category.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_violin_quantity_by_segment(df):
    """Violin plot: quantity distribution by customer segment."""
    if "CustomerSegment" not in df.columns or "Quantity" not in df.columns:
        return None
    plt.figure(figsize=(10,6))
    sns.violinplot(data=df, x="CustomerSegment", y="Quantity", palette=PALETTE)
    plt.title("Quantity Distribution by Customer Segment")
    plt.tight_layout()
    out = os.path.join(VIS_DIR, "quantity_violin_by_segment.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_correlation_heatmap(df):
    """Heatmap: correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="vlag", center=0)
    plt.title("Correlation Matrix (numeric features)")
    plt.tight_layout()
    out = os.path.join(VIS_DIR, "correlation_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out

# -------------------------
# Plotly interactive visualizations
# -------------------------
def interactive_scatter(df):
    """Interactive scatter: Price vs Quantity colored by Category with hover info."""
    if not {"Price","Quantity"}.issubset(df.columns):
        return None
    fig = px.scatter(df, x="Price", y="Quantity", color="Category" if "Category" in df.columns else None,
                     hover_data=["Product","Sales"], title="Price vs Quantity (interactive)")
    out = os.path.join(INT_DIR, "price_quantity_scatter.html")
    fig.write_html(out, include_plotlyjs="cdn")
    return out

def interactive_sunburst(df):
    """Interactive sunburst: sales by Region -> Category -> Product."""
    if not {"Region","Category","Product","Sales"}.issubset(df.columns):
        return None
    agg = df.groupby(["Region","Category","Product"], as_index=False)["Sales"].sum()
    fig = px.sunburst(agg, path=["Region","Category","Product"], values="Sales",
                      color="Category", title="Sales Breakdown: Region → Category → Product")
    out = os.path.join(INT_DIR, "sales_sunburst.html")
    fig.write_html(out, include_plotlyjs="cdn")
    return out

def interactive_time_series(df):
    """Interactive time series with range slider."""
    if "Date" not in df.columns or df["Date"].isna().all():
        return None
    monthly = df.set_index("Date").resample("M")["Sales"].sum().reset_index()
    fig = px.line(monthly, x="Date", y="Sales", title="Monthly Sales (interactive)")
    fig.update_xaxes(rangeslider_visible=True)
    out = os.path.join(INT_DIR, "monthly_sales_interactive.html")
    fig.write_html(out, include_plotlyjs="cdn")
    return out

# -------------------------
# Main orchestration
# -------------------------
def generate_dashboard(csv_path="sales_data.csv"):
    try:
        df = load_data(csv_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    df = preprocess(df)

    # Basic summary
    total_sales = df["Sales"].sum()
    top_product = df.groupby("Product", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False).head(1)
    top_product_name = top_product["Product"].iloc[0] if not top_product.empty else "N/A"
    top_product_sales = top_product["Sales"].iloc[0] if not top_product.empty else 0

    print("=== Sales Summary ===")
    print(f"Total Sales: ₹{int(total_sales):,}")
    print(f"Top Product: {top_product_name} (₹{int(top_product_sales):,})")
    print("=====================")

    # Create static visualizations (Seaborn)
    outputs = []
    outputs.append(plot_total_sales_by_product(df))
    outputs.append(plot_sales_trend(df))
    outputs.append(plot_price_distribution_by_category(df))
    outputs.append(plot_violin_quantity_by_segment(df))
    outputs.append(plot_correlation_heatmap(df))

    # Create interactive visualizations (Plotly)
    outputs.append(interactive_scatter(df))
    outputs.append(interactive_sunburst(df))
    outputs.append(interactive_time_series(df))

    # Report saved files
    print("\nGenerated visualizations (files):")
    for f in outputs:
        if f:
            print(" -", f)
    print("\nStatic images saved to:", VIS_DIR)
    print("Interactive HTML files saved to:", INT_DIR)
    print("\nOpen the HTML files in a browser to explore interactive charts.")

if __name__ == "__main__":
    # Allow optional CSV path argument
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sales_data.csv"
    generate_dashboard(csv_path)
