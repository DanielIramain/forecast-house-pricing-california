#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import Workspace
from evidently.sdk.panels import *
from evidently.legacy.renderers.html_widgets import WidgetSize

def prepare_report():
    # Prepare data and creates a report based on the data drift preset

    data = pd.read_csv('../deployment/output/housing_output.csv')

    train_data = data[:10000]
    val_data = data[10000:]

    num_features = ['housing_median_age', 'median_income', 'real_median_house_value', 'predicted_value', 'diff']
    cat_features = ['model_version']

    data_definition = DataDefinition(numerical_columns=num_features, categorical_columns=cat_features)
    
    train_dataset = Dataset.from_pandas(
        train_data,
        data_definition
    )
    
    val_dataset = Dataset.from_pandas(
        val_data,
        data_definition
    )

    report = Report(metrics=[
        ValueDrift(column='predicted_value'),
        DriftedColumnsCount(),
        MissingValueCount(column='predicted_value'),
    ]
    )

    snapshot = report.run(reference_data=train_dataset, current_data=val_dataset)

    result = snapshot.dict()

    return result, val_data, data_definition

def prepare_dashboard(val_data, data_definition):
    # Prepare a dashboard in the Evidently workspace
    ws = Workspace("workspace")

    project = ws.create_project("California Pricing Quality Project")
    project.description = "Housing project Test"
    project.save()


    regular_report = Report(
        metrics=[
            DataSummaryPreset()
        ],
    )

    data = Dataset.from_pandas(
        val_data,
        data_definition=data_definition,
    )

    regular_snapshot = regular_report.run(current_data=data)

    ws.add_run(project.id, regular_snapshot)

    project.dashboard.add_panel(
        text_panel(title="House pricing in California")
    )

    project.dashboard.add_panel(
        bar_plot_panel(
            title="Inference Count",
            values=[
                PanelMetric(
                    metric="RowCount",
                    legend="count",
                ),
            ],
            size="half",
        ),
    )

    project.dashboard.add_panel(
        line_plot_panel(
            title="Number of Missing Values",
            values=[
                PanelMetric(
                    metric="DatasetMissingValueCount",
                    legend="count"
                ),
            ],
            size="half",
        ),
    )

    project.save()

def run():
    result, val_data, data_definition = prepare_report()
    prepare_dashboard(val_data, data_definition)
    
    return result

if __name__ == "__main__":
    try:
        result = run()
    except Exception as e:
        print(f"Error during monitoring: {e}")