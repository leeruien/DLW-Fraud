"""TrustGuard fraud dashboard (standalone version of Section 11 of TrustGuard_Track2.ipynb).

Loads the trained model.pkl (no training happens here) and starts a Gradio web app where a fraud team can:
  * upload a CSV of transactions and get scores, decisions and three plain-English reasons per alert,
  * move the alert threshold and see the validated catch rate and alert precision live,
  * score a single transaction typed into a form,
  * read the model card.

Usage (from this folder):
    pip install -r requirements.txt
    python dashboard_app.py                      # opens on http://127.0.0.1:7860
    python dashboard_app.py --share              # also prints a public link (useful for a demo)
    python dashboard_app.py --model path/to/model.pkl --test path/to/test.csv --port 7861

In Google Colab:
    !pip install -q gradio
    !python dashboard_app.py --share
"""
import argparse
import glob
import os
import joblib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------------------------
# Command-line options
# ----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='TrustGuard fraud dashboard')
parser.add_argument('--model', default=None, help='path to model.pkl (default: next to this script)')
parser.add_argument('--test', default=None, help='CSV used by the "Try it with the competition test set" button')
parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 7860)))
parser.add_argument('--share', action='store_true', help='create a public Gradio link')
args, _ = parser.parse_known_args()

# ----------------------------------------------------------------------------------------------
# Load the trained model (all cleaning, features and the cutoff are inside it)
# ----------------------------------------------------------------------------------------------
def find_first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

MODEL_PATH = find_first([args.model, os.path.join(HERE, 'model.pkl'), 'model.pkl', '/content/model.pkl'])
if MODEL_PATH is None:
    sys.exit('model.pkl not found. Put it next to dashboard_app.py or pass --model path/to/model.pkl')

TEST_PATH = args.test
if TEST_PATH is None:
    hits = []
    for root in [HERE, os.path.join(HERE, 'Track 2 Dataset'), os.path.join(HERE, '..', 'Track 2 Dataset'), '/content']:
        hits += sorted(glob.glob(os.path.join(root, '*Testing*.csv')))
    TEST_PATH = hits[0] if hits else None

# ----------------------------------------------------------------------------------------------
# Feature engineering (identical to prediction_notebook.ipynb and the training notebook)
# ----------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd

RAW_NUM = ['transaction_amount', 'transaction_hour', 'transactions_last_24h', 'spend_last_24h',
           'account_age', 'new_device', 'transactions_last_1h']
CAT_COLS = ['merchant_category', 'country', 'transaction_channel']
CAT_LEVELS = {
    'merchant_category': ['grocery', 'transport', 'retail', 'entertainment', 'utilities', 'gaming',
                          'food_delivery', 'travel', 'electronics', 'cash_transfer', 'luxury'],
    'country': ['SG', 'MY', 'TH', 'ID', 'VN', 'PH', 'JP', 'US', 'GB', 'AU'],
    'transaction_channel': ['card_present', 'mobile_app', 'ecommerce', 'bank_transfer'],
}
TX1H_LABELS = ['0-2', '3', '4', '5', '6', '7+']
# 0.1% and 99.9% percentiles of the training data (fixed constants, so the function stays stateless)
CLIP = {'transaction_amount': (2.5, 5593.7418), 'spend_last_24h': (0.0, 8906.8026), 'account_age': (5.0, 4000.0),
        'transactions_last_24h': (1.0, 26.0), 'transactions_last_1h': (0.0, 9.0)}
NA_COLS = ['transactions_last_24h', 'spend_last_24h', 'account_age', 'new_device', 'transactions_last_1h']


def make_features(df):
    if isinstance(df, dict):            # a single transaction given as {column: value}
        df = pd.DataFrame([df]) if all(np.isscalar(v) or v is None for v in df.values()) else pd.DataFrame(df)
    elif isinstance(df, pd.Series):
        df = df.to_frame().T
    df = pd.DataFrame(df).copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    df = df.reset_index(drop=True)
    d = pd.DataFrame(index=df.index)
    # cleaning: numbers
    for c in RAW_NUM:
        d[c] = pd.to_numeric(df[c], errors='coerce') if c in df else np.nan
    d = d.astype(float).replace([np.inf, -np.inf], np.nan)
    for c in ['transaction_amount', 'transactions_last_24h', 'spend_last_24h', 'account_age', 'transactions_last_1h']:
        d[c] = d[c].clip(lower=0)
    d['transaction_hour'] = np.floor(d['transaction_hour'] % 24)
    d['new_device'] = d['new_device'].clip(0, 1).round()
    for c, (lo, hi) in CLIP.items():
        d[c] = d[c].clip(lo, hi)
    d['transactions_last_24h'] = np.fmax(d['transactions_last_24h'], d['transactions_last_1h'])
    # cleaning: categories (unseen or missing -> 'unknown', which every encoder treats as all-zero)
    for c in CAT_COLS:
        s = df[c] if c in df else pd.Series(np.nan, index=df.index)
        s = s.astype('string').str.strip()
        s = s.str.upper() if c == 'country' else s.str.lower().str.replace(' ', '_')
        allowed = set(CAT_LEVELS[c])
        d[c] = [v if v in allowed else 'unknown' for v in s.to_numpy(dtype=object, na_value=None)]
    known = lambda s: s.notna()
    # velocity
    d['vel_mult'] = np.log1p(d.transactions_last_1h / (d.transactions_last_24h / 24 + 0.1))
    d['tx1h_share'] = d.transactions_last_1h / (d.transactions_last_24h + 1)
    d['burst'] = (d.transactions_last_1h >= 4).astype(float).where(known(d.transactions_last_1h))
    band = pd.cut(d.transactions_last_1h, [-np.inf, 2, 3, 4, 5, 6, np.inf], labels=TX1H_LABELS).astype(object)
    d['tx1h_band'] = band.where(band.notna(), '0-2').astype(str)
    # monetary
    d['log_amount'] = np.log1p(d.transaction_amount)
    d['log_spend24'] = np.log1p(d.spend_last_24h)
    d['log_tx24'] = np.log1p(d.transactions_last_24h)
    avg24 = d.spend_last_24h / (d.transactions_last_24h + 1)
    d['amt_vs_avg24'] = d.log_amount - np.log1p(avg24)
    d['amt_share'] = d.transaction_amount / (d.spend_last_24h + d.transaction_amount + 1)
    d['amt_vs_age'] = d.log_amount - np.log1p(d.account_age)
    d['spend_intensity'] = d.log_spend24 - np.log1p(d.account_age)
    # identity and trust
    d['log_age'] = np.log1p(d.account_age)
    d['young'] = (d.account_age < 90).astype(float).where(known(d.account_age))
    d['nd_young'] = d.new_device * d.young
    # context
    d['hour_sin'] = np.sin(2 * np.pi * d.transaction_hour / 24)
    d['hour_cos'] = np.cos(2 * np.pi * d.transaction_hour / 24)
    d['night'] = (d.transaction_hour <= 5).astype(float).where(known(d.transaction_hour))
    d['remote'] = d.transaction_channel.isin(['ecommerce', 'mobile_app', 'bank_transfer']).astype(float)
    d['liquid'] = d.merchant_category.isin(['luxury', 'electronics', 'cash_transfer']).astype(float)
    d['foreign'] = ((d.country != 'SG') & (d.country != 'unknown')).astype(float)
    # data quality
    for c in NA_COLS:
        d[c + '_na'] = d[c].isna().astype(float)
    return d


# Column lists used by the models
LR_NUM = ['log_amount', 'log_spend24', 'log_age', 'transactions_last_24h', 'new_device', 'hour_sin', 'hour_cos',
          'amt_vs_avg24', 'amt_share', 'vel_mult', 'tx1h_share', 'night', 'young', 'nd_young',
          'transactions_last_24h_na', 'spend_last_24h_na', 'account_age_na', 'new_device_na', 'transactions_last_1h_na',
          'log_tx24']
LR_CAT = CAT_COLS + ['tx1h_band']
TREE_NUM = RAW_NUM + ['amt_vs_avg24', 'amt_share', 'vel_mult', 'tx1h_share', 'amt_vs_age', 'spend_intensity']
TREE_CAT = CAT_COLS
ALL_FEATURES = sorted(set(LR_NUM + LR_CAT + TREE_NUM + TREE_CAT))

LR_LEVELS = [CAT_LEVELS['merchant_category'], CAT_LEVELS['country'], CAT_LEVELS['transaction_channel'], TX1H_LABELS]
RAW_ORDER = ['transaction_amount', 'transaction_hour', 'merchant_category', 'country', 'transaction_channel',
             'transactions_last_24h', 'spend_last_24h', 'account_age', 'new_device', 'transactions_last_1h']

def to_frame(data):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    df = pd.DataFrame(data).copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    return df.reset_index(drop=True)

def concept_of(col):
    if col.startswith('tx1h_band') or col in ('vel_mult', 'tx1h_share'):
        return 'velocity_1h'
    if col in ('log_age', 'young'):
        return 'account_age'
    if col in ('new_device', 'nd_young'):
        return 'new_device'
    if col in ('log_amount', 'amt_vs_avg24', 'amt_share'):
        return 'amount'
    if col == 'log_spend24':
        return 'spend_24h'
    if col in ('transactions_last_24h', 'log_tx24'):
        return 'activity_24h'
    if col in ('hour_sin', 'hour_cos', 'night'):
        return 'time_of_day'
    if col.endswith('_na'):
        return 'missing_data'
    for c in CAT_COLS:
        if col.startswith(c):
            return c
    return col

def reason(concept, r):
    f = lambda v, fmt: 'unknown' if pd.isna(v) else fmt.format(v)
    if concept == 'velocity_1h':
        return f'{f(r.transactions_last_1h, "{:.0f}")} transactions in the last hour (typical: 0 to 2)'
    if concept == 'account_age':
        return f'account is only {f(r.account_age, "{:,.0f}")} days old'
    if concept == 'new_device':
        return 'first transaction from this device'
    if concept == 'amount':
        avg = r.spend_last_24h / (r.transactions_last_24h + 1) if pd.notna(r.spend_last_24h) else np.nan
        ratio = r.transaction_amount / avg if pd.notna(avg) and avg > 0 else np.nan
        extra = f' ({ratio:.1f}x the account average today)' if pd.notna(ratio) else ''
        return f'large amount {f(r.transaction_amount, "{:,.2f}")}{extra}'
    if concept == 'spend_24h':
        return f'spent {f(r.spend_last_24h, "{:,.0f}")} in the last 24h'
    if concept == 'activity_24h':
        return f'{f(r.transactions_last_24h, "{:.0f}")} transactions in the last 24h'
    if concept == 'time_of_day':
        return f'made at {f(r.transaction_hour, "{:02.0f}")}:00 (night hours carry more fraud)'
    if concept == 'merchant_category':
        return f'higher-risk merchant category ({r.merchant_category})'
    if concept == 'country':
        return f'higher-risk country ({r.country})'
    if concept == 'transaction_channel':
        return f'remote channel ({r.transaction_channel})'
    if concept == 'missing_data':
        return 'key account fields are missing'
    return concept

class Model:
    # model.pkl = (scikit-learn pipeline, feature list); facts such as the cutoff are stored on the pipeline.
    def __init__(self, path):
        self.pipe, self.features = joblib.load(path)
        self.threshold = float(getattr(self.pipe, 'threshold_', 0.5))
        self.train_rate_ = float(getattr(self.pipe, 'train_rate_', 0.0176))
        self.op_curve_ = getattr(self.pipe, 'op_curve_')
        self.model_name = getattr(self.pipe, 'model_name', 'Capped logistic regression')
        self.cv_pr_auc = float(getattr(self.pipe, 'cv_pr_auc', float('nan')))
        self.cv_shift_pr_auc = float(getattr(self.pipe, 'cv_shift_pr_auc', float('nan')))
        self.feature_names_ = list(LR_NUM) + [f'{c}={l}' for c, levels in zip(LR_CAT, LR_LEVELS) for l in levels]

    def predict_proba(self, data):
        F = make_features(data)
        return self.pipe.predict_proba(F[self.features]) if len(F) else np.zeros((0, 2))

    def explain(self, data, top_k=3):
        import types
        F = make_features(data)
        D = self.pipe.named_steps['prep'].transform(F[self.features])
        contrib = (D - self.pipe.design_mean_) * self.pipe.named_steps['lr'].coef_.ravel()
        concepts = sorted(set(concept_of(c) for c in self.feature_names_))
        member = np.array([[concept_of(c) == k for k in concepts] for c in self.feature_names_], dtype=float)
        G = contrib @ member
        order = np.argsort(-G, axis=1)[:, :top_k]
        rows = [types.SimpleNamespace(**r) for r in F.to_dict('records')]
        return [' | '.join(reason(concepts[j], r) for j in order[i] if G[i, j] > 0.05) or 'no strong risk factors'
                for i, r in enumerate(rows)]

final_model = Model(MODEL_PATH)

# ----------------------------------------------------------------------------------------------
# Chart style
# ----------------------------------------------------------------------------------------------
SURFACE, INK, INK2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
BLUE = '#2a78d6'
GOOD, WARNING, CRITICAL = '#0ca30c', '#fab219', '#d03b3b'
matplotlib.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.titlecolor': INK, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
})

OP_CURVE = pd.DataFrame(final_model.op_curve_)          # validation PR curve stored inside model.pkl
THRESHOLD = final_model.threshold
BASE_RATE = final_model.train_rate_
BLOCK_THRESHOLD = float(OP_CURVE.loc[OP_CURVE.precision >= 0.6, 'threshold'].min()) if (OP_CURVE.precision >= 0.6).any() else 0.5

# ----------------------------------------------------------------------------------------------
# Scoring, decisions and charts
# ----------------------------------------------------------------------------------------------
def validation_at(t):
    r = OP_CURVE.iloc[(OP_CURVE.threshold - t).abs().argmin()]
    return float(r.precision), float(r.recall)

def score_frame(df, model=final_model):
    out = to_frame(df)
    p = model.predict_proba(out)[:, 1]
    out.insert(0, 'fraud_probability', np.round(p, 5))
    out.insert(1, 'reasons', model.explain(out, top_k=3))
    return out

def decisions(p, t):
    return np.where(p >= max(BLOCK_THRESHOLD, t), 'block', np.where(p >= t, 'review', 'allow'))

def schema_notes(df):
    cols = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    missing_cols = [c for c in RAW_ORDER if c not in cols]
    notes = [f'Missing columns filled as unknown: {", ".join(missing_cols)}'] if missing_cols else []
    na = to_frame(df)[[c for c in RAW_ORDER if c in cols]].isna().mean()
    if (na > 0).any():
        notes.append('Columns with gaps (numbers filled with training medians, categories treated as unknown): '
                     + ', '.join(f'{c} {v:.1%}' for c, v in na[na > 0].items()))
    return notes

def dashboard_figures(scored, t, segment):
    d = scored.assign(decision=decisions(scored.fraud_probability.to_numpy(), t))
    fig1, ax = plt.subplots(figsize=(5.2, 3.2))
    counts = d.decision.value_counts().reindex(['allow', 'review', 'block']).fillna(0)
    bars = ax.bar(counts.index, counts.values, color=[GOOD, WARNING, CRITICAL], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{int(v):,}', ha='center', va='bottom', fontsize=9, color=INK2)
    ax.set_yscale('log'); ax.set_ylim(bottom=1); ax.set_ylabel('transactions (log scale)')
    ax.set_title('Decisions at this threshold', fontsize=10)
    fig1.tight_layout(); plt.close(fig1)
    seg = segment if segment in d.columns else 'merchant_category'
    g = d.assign(alert=d.decision != 'allow').groupby(d[seg].fillna('unknown'))['alert'].agg(['mean', 'size']).sort_values('mean')
    fig2, ax = plt.subplots(figsize=(5.2, 3.2 + 0.12 * len(g)))
    ax.barh(g.index.astype(str), g['mean'] * 100, color=BLUE, height=0.6)
    ax.set_xlabel('alert rate (%)'); ax.set_title(f'Alert rate by {seg.replace("_", " ")}', fontsize=10)
    fig2.tight_layout(); plt.close(fig2)
    return fig1, fig2, d

def kpi_markdown(d, t, notes):
    n, alerts = len(d), int((d.decision != 'allow').sum())
    vp, vr = validation_at(t)
    lines = [f'### {n:,} transactions scored  |  {alerts:,} alerts ({alerts / max(n, 1):.1%})  |  '
             f'expected frauds in this file: {d.fraud_probability.sum():,.0f}',
             f'**At threshold {t:.3f}**, on held-out validation data the model caught **{vr:.0%} of frauds**, and '
             f'**{vp:.0%} of its alerts were real fraud** ({vp / BASE_RATE:.0f}x better than random checking). '
             f'Transactions at or above {max(BLOCK_THRESHOLD, t):.2f} are marked *block* (validation precision at least 60%).']
    lines += [f'> {x}' for x in notes]
    return '\n\n'.join(lines)

def render(scored, t, segment, notes=()):
    fig1, fig2, d = dashboard_figures(scored, t, segment)
    show = ['fraud_probability', 'decision', 'reasons', 'id', 'transaction_amount', 'merchant_category', 'country',
            'transaction_channel', 'transactions_last_1h', 'account_age', 'new_device']
    top = d.sort_values('fraud_probability', ascending=False).reindex(columns=show).head(25)
    path = os.path.abspath('trustguard_scored.csv')
    d.to_csv(path, index=False)
    return kpi_markdown(d, t, notes), fig1, fig2, top, path

def run_batch(file_path, t, segment):
    if file_path is None:
        return 'Upload a CSV to begin.', None, None, None, None, None
    try:
        raw = pd.read_csv(file_path)
    except Exception as e:
        return f'Could not read the file: {e}', None, None, None, None, None
    scored = score_frame(raw)
    return (*render(scored, t, segment, schema_notes(raw)), scored)

def score_one(amount, hour, merchant, country, channel, tx24, spend24, age, new_dev, tx1h, t):
    row = dict(transaction_amount=amount, transaction_hour=hour, merchant_category=merchant, country=country,
               transaction_channel=channel, transactions_last_24h=tx24, spend_last_24h=spend24, account_age=age,
               new_device=1 if new_dev else 0, transactions_last_1h=tx1h)
    p = float(final_model.predict_proba(row)[0, 1])
    dec = decisions(np.array([p]), t)[0]
    return (f'## {dec.upper()}: fraud probability {p:.1%}\n\n{p / BASE_RATE:.1f} times the average transaction risk ({BASE_RATE:.1%}).\n\n'
            f'**Why:** {final_model.explain(row)[0]}')

MODEL_CARD = f'''
### Model card
* **Model:** {final_model.model_name}: logistic regression on {len(final_model.feature_names_)} engineered behavioural inputs, with extreme values capped so the shifted test population cannot dominate a score.
* **Training data:** 20,000 transactions, 353 frauds (1.76%). No resampling, so probabilities are calibrated to that rate.
* **Validation:** 5-fold stratified CV repeated 3 times. PR-AUC {final_model.cv_pr_auc:.3f} (random = 0.018), shift-weighted PR-AUC {final_model.cv_shift_pr_auc:.3f}.
* **Default threshold:** {THRESHOLD:.3f} (maximises F2, recall weighted twice).
* **Known limits:** the test population differs from training (larger amounts in AU, JP and MY, older accounts, more new devices). No account ids, so no per-customer history. Fraud labels are scarce, so scores rank risk. They are not proof of fraud. Always keep a human in the loop for blocks.
'''

# ----------------------------------------------------------------------------------------------
# The app
# ----------------------------------------------------------------------------------------------
def build_app():
    import gradio as gr
    with gr.Blocks(title='TrustGuard fraud dashboard') as demo:
        gr.Markdown('# TrustGuard: transaction fraud screening\nUpload transactions, choose how strict to be, and review the riskiest cases with reasons.')
        with gr.Tab('Batch scoring'):
            with gr.Row():
                file_in = gr.File(label='Transactions CSV (same columns as the competition data)', file_types=['.csv'], type='filepath')
                with gr.Column():
                    t_slider = gr.Slider(0.01, 0.6, value=round(THRESHOLD, 3), step=0.005, label='Alert threshold (lower = catch more, more false alarms)')
                    seg = gr.Dropdown(['merchant_category', 'country', 'transaction_channel'], value='merchant_category', label='Break down alerts by')
                    sample_btn = gr.Button('Try it with the competition test set', visible=TEST_PATH is not None)
            kpis = gr.Markdown()
            with gr.Row():
                plot1 = gr.Plot(label='Decisions')
                plot2 = gr.Plot(label='Alert rate by segment')
            table = gr.Dataframe(label='Riskiest 25 transactions, with reasons', wrap=True, column_widths=['8%', '7%', '45%'] + ['5%'] * 8)
            download = gr.File(label='Download all scored transactions')
            state = gr.State()
            file_in.change(run_batch, [file_in, t_slider, seg], [kpis, plot1, plot2, table, download, state])
            sample_btn.click(lambda t, s: run_batch(TEST_PATH, t, s), [t_slider, seg], [kpis, plot1, plot2, table, download, state])
            for ctl in (t_slider, seg):
                ctl.change(lambda sc, t, s: render(sc, t, s) if sc is not None else ('Upload a CSV to begin.', None, None, None, None),
                           [state, t_slider, seg], [kpis, plot1, plot2, table, download])
        with gr.Tab('Single transaction'):
            with gr.Row():
                with gr.Column():
                    a = gr.Number(250, label='Transaction amount')
                    h = gr.Slider(0, 23, value=2, step=1, label='Hour of day')
                    mc = gr.Dropdown(CAT_LEVELS['merchant_category'], value='electronics', label='Merchant category')
                    co = gr.Dropdown(CAT_LEVELS['country'], value='SG', label='Country')
                    ch = gr.Dropdown(CAT_LEVELS['transaction_channel'], value='ecommerce', label='Channel')
                with gr.Column():
                    t24 = gr.Number(8, label='Transactions in last 24h')
                    s24 = gr.Number(900, label='Spend in last 24h')
                    ag = gr.Number(25, label='Account age (days)')
                    nd = gr.Checkbox(True, label='New device')
                    t1 = gr.Number(5, label='Transactions in last hour')
                    btn = gr.Button('Score transaction', variant='primary')
            verdict = gr.Markdown()
            btn.click(score_one, [a, h, mc, co, ch, t24, s24, ag, nd, t1, t_slider], verdict)
        with gr.Tab('Model card'):
            gr.Markdown(MODEL_CARD)
    return demo

if __name__ == '__main__':
    try:
        import gradio  # noqa: F401
    except ImportError:
        sys.exit('gradio is not installed. Run: pip install gradio')
    print(f'Model: {getattr(final_model, "model_name", type(final_model).__name__)} | cutoff {THRESHOLD:.4f} | loaded from {MODEL_PATH}')
    print(f'Sample test file: {TEST_PATH or "none found (the sample button is hidden)"}')
    build_app().launch(share=args.share or 'google.colab' in sys.modules, server_port=args.port)
