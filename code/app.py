# -*- coding: utf-8 -*-
"""
K-Means Anomaly Detector — Interactive Dashboard (Plotly + Dash)
Optimized for Render Free Tier (Memory Constraints)
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ---------------- Paths ----------------
TRAIN_PATH  = r"KDDTrain+.txt"
VAL_PATH    = r"KDDTest+.txt"
TEST21_PATH = r"KDDTest-21.txt"

# ---------------- Helpers ----------------
def load_df(path):
    # تم تحديد nrows=20000 لتقليل استهلاك الذاكرة لتناسب الخطة المجانية (512MB)
    df = pd.read_csv(path, header=None, nrows=20000)
    num_features = df.shape[1] - 1
    cols = [f'feature_{i}' for i in range(num_features)] + ['label']
    df.columns = cols
    return df

def extract_actual(df, label_col):
    col = df[label_col].astype(str)
    norm = col.str.strip().str.rstrip('.').str.lower()
    if norm.str.contains(r'\bnormal\b', regex=True).any():
        df['Actual'] = norm.str.contains(r'\bnormal\b', regex=True).apply(lambda x: 0 if x else 1)
    else:
        numeric = pd.to_numeric(col, errors='coerce')
        if numeric.notna().any():
            uniq = np.unique(numeric[~np.isnan(numeric)])
            if set(uniq) <= {0,1}:
                df['Actual'] = numeric.astype(int).apply(lambda x: 0 if x==0 else 1)
            else:
                mode = numeric.mode().iloc[0]
                df['Actual'] = numeric.apply(lambda x: 0 if x==mode else 1)
        else:
            df['Actual'] = 1
    return df

def detect_categorical_columns(df, exclude_cols):
    return [c for c in df.columns if df[c].dtype == 'object' and c not in exclude_cols]

def encode_train(df, categorical_cols):
    encoders = {}
    for c in categorical_cols:
        le = LabelEncoder()
        df[c] = df[c].fillna('___nan___').astype(str)
        le.fit(df[c])
        df[c] = le.transform(df[c])
        encoders[c] = le
    return encoders

def encode_test(df, categorical_cols, encoders):
    for c in categorical_cols:
        le = encoders[c]
        df[c] = df[c].fillna('___nan___').astype(str)
        mapping = {cls:i for i,cls in enumerate(le.classes_)}
        df[c] = df[c].map(lambda x: mapping.get(x, len(le.classes_)))
    return df

def build_feature_matrix(df, exclude_cols):
    X_all = df.drop(columns=exclude_cols + ['Actual'])
    numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    return X_all[numeric_cols], numeric_cols

def eval_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'cm':cm, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1}

def compute_distances(df, X_scaled, centroids, threshold):
    dists = np.min(cdist(X_scaled, centroids), axis=1)
    df['dist_to_centroid'] = dists
    df['Anomaly'] = (dists > threshold).astype(int)
    return df, dists

# ---------------- Load & Preprocess ----------------
train_df = load_df(TRAIN_PATH)
label_source = 'feature_41'
train_df = extract_actual(train_df, label_source)
exclude_columns = [label_source, 'label'] if label_source != 'label' else ['label']

categorical_cols = detect_categorical_columns(train_df, exclude_cols=exclude_columns)
encoders = encode_train(train_df, categorical_cols) if categorical_cols else {}

X_train, used_cols = build_feature_matrix(train_df, exclude_cols=exclude_columns)
y_train = train_df['Actual'].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ---------------- KMeans (Optimized for Server) ----------------
# تم تثبيت قيمة k وتخفيف إعدادات التدريب لتسريع الإقلاع وتجنب استهلاك الذاكرة
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=5).fit(X_train_scaled)
centroids = kmeans.cluster_centers_

dists_train = np.min(cdist(X_train_scaled, centroids), axis=1)
train_df['dist_to_centroid'] = dists_train
train_df['Cluster'] = kmeans.labels_

# Validation
val_df = load_df(VAL_PATH)
val_df = extract_actual(val_df, label_source if label_source in val_df.columns else 'label')
if categorical_cols:
    val_df = encode_test(val_df, categorical_cols, encoders)
X_val = val_df.drop(columns=[label_source,'label','Actual'], errors='ignore')[used_cols].astype(float)
X_val_scaled = scaler.transform(X_val)
val_df['Cluster'] = kmeans.predict(X_val_scaled)

# Test21
test21_df = load_df(TEST21_PATH)
test21_df = extract_actual(test21_df, label_source if label_source in test21_df.columns else 'label')
if categorical_cols:
    test21_df = encode_test(test21_df, categorical_cols, encoders)
X_test = test21_df.drop(columns=[label_source,'label','Actual'], errors='ignore')[used_cols].astype(float)
X_test_scaled = scaler.transform(X_test)
test21_df['Cluster'] = kmeans.predict(X_test_scaled)

# ---------------- PCA for 2D Visualization ----------------
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# Default threshold and Initial Distance Calculation
best_th = np.percentile(dists_train, 90)
train_df, _ = compute_distances(train_df, X_train_scaled, centroids, best_th)
val_df, _   = compute_distances(val_df, X_val_scaled, centroids, best_th)
test21_df, _ = compute_distances(test21_df, X_test_scaled, centroids, best_th)

# ---------------- Dash App ----------------
app = dash.Dash(__name__)
server = app.server # مهم جداً لسيرفر Gunicorn

# Helper functions for plotting
def make_hist(df):
    return go.Histogram(
        x=df['dist_to_centroid'],
        nbinsx=200,
        opacity=0.6,
        marker_color='blue'
    )

def make_scatter(df, X_pca):
    cluster_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
    colors = []
    for a,c in zip(df['Anomaly'], df['Cluster']):
        if a==1:
            colors.append('red')
        else:
            colors.append(cluster_colors[c % len(cluster_colors)])
    return go.Scatter(
        x=X_pca[:,0],
        y=X_pca[:,1],
        mode='markers',
        marker=dict(color=colors, size=6, opacity=0.7),
        text=[f"Cluster: {c}<br>Anomaly: {a}" for c,a in zip(df['Cluster'], df['Anomaly'])],
        hoverinfo='text'
    )

def make_pie(df):
    counts = df['Anomaly'].value_counts()
    return go.Pie(
        labels=['Normal','Anomaly'],
        values=[counts.get(0,0), counts.get(1,0)],
        hole=0.4
    )

def make_metrics_table(df_list, names):
    metrics_list = []
    for name, df in zip(names, df_list):
        m = eval_metrics(df['Actual'], df['Anomaly'])
        metrics_list.append({
            'Dataset': name,
            'Accuracy': f"{m['acc']:.3f}",
            'Precision': f"{m['prec']:.3f}",
            'Recall': f"{m['rec']:.3f}",
            'F1-Score': f"{m['f1']:.3f}"
        })
    return pd.DataFrame(metrics_list)

metrics_df_init = make_metrics_table([train_df, val_df, test21_df], ['Train','Validation','Test21'])

# ---------------- Layout ----------------
app.layout = html.Div(style={'padding':'20px','font-family':'Arial'}, children=[
    html.H2("K-Means Anomaly Detection Dashboard", style={'textAlign':'center'}),
    html.H4(f"Selected k={best_k}", style={'textAlign':'center'}),

    html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label':'Train','value':'Train'},
                     {'label':'Validation','value':'Validation'},
                     {'label':'Test21','value':'Test21'}],
            value='Train'
        ),
        html.Label("Threshold Percentile:"),
        dcc.Slider(id='threshold-slider', min=50, max=99, step=1, value=90,
                   marks={i:f"{i}%" for i in range(50,100,5)})
    ], style={'width':'50%', 'margin':'20px auto'}),

    dcc.Graph(id='histogram-graph'),
    dcc.Graph(id='scatter-pca'),
    dcc.Graph(id='pie-anomaly'),

    html.H4("Metrics Summary", style={'margin-top':'30px', 'textAlign':'center'}),
    dash_table.DataTable(
        id='metrics-table',
        columns=[{"name": i, "id": i} for i in metrics_df_init.columns],
        data=metrics_df_init.to_dict('records'),
        style_cell={'textAlign':'center', 'padding':'5px'},
        style_header={'backgroundColor':'lightblue','fontWeight':'bold'},
        style_data={'backgroundColor':'beige'},
        style_table={'width':'60%','margin':'0 auto'}
    )
])

# ---------------- Callbacks ----------------
@app.callback(
    [Output('histogram-graph','figure'),
     Output('scatter-pca','figure'),
     Output('pie-anomaly','figure'),
     Output('metrics-table', 'data')],
    [Input('dataset-dropdown','value'),
     Input('threshold-slider','value')]
)
def update_dashboard(dataset_name, threshold_percentile):
    
    th_train = np.percentile(train_df['dist_to_centroid'], threshold_percentile)
    train_df['Anomaly'] = (train_df['dist_to_centroid'] > th_train).astype(int)
    
    th_val = np.percentile(val_df['dist_to_centroid'], threshold_percentile)
    val_df['Anomaly'] = (val_df['dist_to_centroid'] > th_val).astype(int)
    
    th_test = np.percentile(test21_df['dist_to_centroid'], threshold_percentile)
    test21_df['Anomaly'] = (test21_df['dist_to_centroid'] > th_test).astype(int)

    updated_metrics_df = make_metrics_table(
        [train_df, val_df, test21_df], 
        ['Train', 'Validation', 'Test21']
    )
    table_data = updated_metrics_df.to_dict('records')

    if dataset_name=='Train':
        df = train_df.copy()
        X_pca = X_train_pca
        threshold_value = th_train
    elif dataset_name=='Validation':
        df = val_df.copy()
        X_pca = X_val_pca
        threshold_value = th_val
    else:
        df = test21_df.copy()
        X_pca = X_test_pca
        threshold_value = th_test

    hist_fig = go.Figure()
    hist_fig.add_trace(make_hist(df))
    hist_fig.add_shape({'type':'line','x0':threshold_value,'x1':threshold_value,'y0':0,'y1':df.shape[0],
                        'line':{'color':'red','width':3,'dash':'dashdot'}})
    hist_fig.update_layout(title=f"Histogram: Distances to Centroid ({dataset_name})",
                           xaxis_title='Distance', yaxis_title='Count')

    scatter_fig = go.Figure()
    scatter_fig.add_trace(make_scatter(df, X_pca))
    scatter_fig.update_layout(title=f"PCA 2D Cluster Visualization ({dataset_name})",
                              xaxis_title='PCA1', yaxis_title='PCA2')

    pie_fig = go.Figure()
    pie_fig.add_trace(make_pie(df))
    pie_fig.update_layout(title=f"Anomaly Proportion ({dataset_name})")

    return hist_fig, scatter_fig, pie_fig, table_data

if __name__ == '__main__':
    app.run(debug=True)