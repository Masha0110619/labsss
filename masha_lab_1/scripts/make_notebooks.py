from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def nb(cells):
    book = nbf.v4.new_notebook()
    book["cells"] = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    book["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return book


SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
warnings.filterwarnings("ignore", message="Bins whose width are too small")
pd.set_option("display.width", 160)
df = pd.read_csv(ROOT / "data/processed/ecommerce_churn_mar.csv")
manifest = json.loads((ROOT / "data/processed/injection_manifest.json").read_text(encoding="utf-8"))
df.shape, manifest["churn_share"]"""

EDA = [
    ("md", "# 00 — Розвідувальний аналіз: E Commerce Dataset (відтік клієнтів)\n\nДомен E-commerce, ТЗ ЛР №1 (розділ 4). Захищені ознаки: `CityTier` (географія, основна) і `Gender`; `PreferredLoginDevice` — потенційний проксі. Аналіз лише описовий: жодних статистик, які потім потраплять у конвеєр, тут не обчислюється й не зберігається."),
    ("code", SETUP),
    ("md", "## Джерело, очищення міток і пропуски\n\nСирий файл містить дублікати міток однієї категорії (`Phone`/`Mobile Phone`, `CC`/`Credit Card`, `COD`/`Cash on Delivery`, `Mobile`/`Mobile Phone`) — зведено в `scripts/prepare_data.py`. Природні пропуски — лише ≤ 5.5 % у семи колонках, тож для демонстрації другої і третьої гілок політики додано MAR-пропуски, залежні від спостережуваних незахищених ознак, і одну структурну відсутність."),
    ("code", "pd.DataFrame({'до': manifest['missing_share_before'], 'після': manifest['missing_share_after']}).fillna(0).sort_values('після', ascending=False)"),
    ("code", "print(json.dumps(manifest['mar_rules'] + [manifest['structural_rule']], ensure_ascii=False, indent=2))"),
    ("md", "## Цільова змінна та базові показники у захищених групах\n\nЧастки відтоку між містами різного рівня і між статями відрізняються — це визначає вибір між DPR і EOR (контрольне питання блоку В)."),
    ("code", "print('churn:', df['Churn'].mean().round(4))\nfor p in ['CityTier', 'Gender', 'PreferredLoginDevice']:\n    display(df.groupby(p)['Churn'].agg(['mean', 'size']).round(3))"),
    ("md", "## Розподіли числових ознак\n\nАсиметрія (`CashbackAmount`, `WarehouseToHome`, `NumberOfAddress`) — аргумент для Yeo–Johnson у чутливій до масштабу гілці та для квантильних бінів у інваріантній."),
    ("code", "num = ['Tenure','WarehouseToHome','CashbackAmount','OrderAmountHikeFromlastYear','DaySinceLastOrder','HourSpendOnApp','NumberOfAddress','OrderCount']\naxes = df[num].hist(bins=30, figsize=(14, 7), layout=(2, 4))\nplt.tight_layout(); plt.show()\ndf[num].skew().round(2)"),
    ("md", "## Категоріальні ознаки після очищення"),
    ("code", "for c in ['PreferredLoginDevice','PreferredPaymentMode','PreferedOrderCat','MaritalStatus','SatisfactionScore']:\n    print(c, df[c].value_counts(dropna=False).to_dict())"),
    ("md", "## Проксі захищених ознак\n\nV Крамера між `PreferredLoginDevice` і захищеними ознаками, а також між захищеними ознаками між собою. Число з `artifacts/analysis.json` (скрипт `scripts/analyze.py`) — той самий розрахунок для всіх ознак."),
    ("code", "from scipy.stats import chi2_contingency\ndef cramers_v(a, b):\n    t = pd.crosstab(a, b); chi2 = chi2_contingency(t, correction=False)[0]\n    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))\npairs = [('PreferredLoginDevice','CityTier'), ('PreferredLoginDevice','Gender'), ('CityTier','Gender'), ('PreferredPaymentMode','CityTier'), ('PreferedOrderCat','Gender')]\npd.DataFrame([(a, b, round(cramers_v(df[a].fillna('NA'), df[b]), 3)) for a, b in pairs], columns=['ознака','захищена','V'])"),
]

QG = [
    ("md", "# 01 — Quality Gate 1: ворота якості, евристичний бейзлайн, аудиторський слід\n\nДемонстрація вимог п. 3.4.9 ТЗ: спрацювання `validate_config` на зіпсованій конфігурації, збирання без звернення до даних, навчання адаптивного імпутера лише на train, аудиторський слід, перевірка схеми ключів `scale_sensitive / scale_invariant` тестом на множення колонки на константу."),
    ("code", SETUP),
    ("md", "## 1. Евристичний бейзлайн (без навчання, на всьому наборі)\n\nПравило: клієнт у зоні відтоку, якщо він з нами ≤ 1 місяць **або** подавав скаргу і при цьому був активний в останні 3 дні (скарга після нещодавнього замовлення). Метрики рахуються для класу `Churn = 1`."),
    ("code", "from sklearn.metrics import accuracy_score, recall_score, precision_score\ny = df['Churn']\nrule = (df['Tenure'].le(1) | (df['Complain'].eq(1) & df['DaySinceLastOrder'].le(3))).fillna(False).astype(int)\nprint('Accuracy:', round(accuracy_score(y, rule), 3))\nprint('Recall(churn):', round(recall_score(y, rule), 3))\nprint('Precision(churn):', round(precision_score(y, rule), 3))\nprint('частка позначених:', round(rule.mean(), 3), '| тривіальне «ніхто не піде»: Accuracy', round(1 - y.mean(), 3))"),
    ("md", "## 2. Спрацювання воріт якості\n\nУ `tests/fixtures/config_protected_leak.yaml` захищена ознака `CityTier` внесена до `features`. Конструктор має зупинитись до збирання конвеєра."),
    ("code", "from src.pipeline_builder import DynamicPipelineBuilder, ConfigError, audit_trail\ntry:\n    DynamicPipelineBuilder(str(ROOT / 'tests/fixtures/config_protected_leak.yaml'))\nexcept ValueError as e:\n    print('ValueError:', e)"),
    ("md", "Друга перевірка — структурна: невідомий крок у `transform_policy` або неупорядкована політика імпутації також зупиняють конструктор (`ConfigError`), тож помилка конфігурації не доживає до навчання."),
    ("code", "import yaml, tempfile\ncfg = yaml.safe_load(open(ROOT / 'config/config_lr.yaml', encoding='utf-8'))\ncfg['transform_policy']['scale_sensitive']['numeric_discrete'] = ['magic_scaler']\nwith tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False, encoding='utf-8') as f:\n    yaml.safe_dump(cfg, f, allow_unicode=True)\ntry:\n    DynamicPipelineBuilder(f.name)\nexcept ConfigError as e:\n    print('ConfigError:', e)"),
    ("md", "## 3. Збирання без даних, навчання лише на train\n\n`build()` повертає ненавчений `ColumnTransformer`; розбиття 70/30 (stratify, seed 42) — перша випадковість у курсі з'являється саме тут, але лише для демонстрації: класифікатори не навчаються."),
    ("code", "from sklearn.model_selection import train_test_split\nfrom sklearn.utils.validation import check_is_fitted\nfrom sklearn.exceptions import NotFittedError\nX = df.drop(columns=['Churn']); y = df['Churn']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)\nct = DynamicPipelineBuilder(str(ROOT / 'config/config_lr.yaml')).build()\ntry:\n    check_is_fitted(ct); print('навчений?!')\nexcept NotFittedError:\n    print('build() повернув ненавчений ColumnTransformer')\nZ_train = ct.fit_transform(X_train, y_train); Z_test = ct.transform(X_test)\nprint('train', Z_train.shape, 'test', Z_test.shape, '| NaN у виході:', int(np.isnan(np.asarray(Z_train, float)).sum()))"),
    ("md", "## 4. Аудиторський слід (п. 3.4.9, крок 4)\n\nЧастки пропусків обчислені імпутером **на тренувальній підвибірці** і відрізняються від часток у повному наборі. У гілці `scale_sensitive` політика диференційована: індикатор створюється навіть при малій частці пропусків, бо медіанна заміна зміщує розподіл для агрегуючої моделі."),
    ("code", "tr = audit_trail(ct)\ntr"),
    ("code", "ct_inv = DynamicPipelineBuilder(str(ROOT / 'config/config_rf.yaml')).build().fit(X_train, y_train)\ntr_inv = audit_trail(ct_inv)\ncmp = tr[['group','missing_share','rule_index','strategy','indicator']].rename(columns={'indicator':'indicator_sensitive'})\ncmp['indicator_invariant'] = tr_inv['indicator']\ncmp['rules_shown'] = cmp['rule_index'].map({0:'≤10%', 1:'10–40%', 2:'>40%'})\ncmp"),
    ("md", "Продемонстровані гілки: числові — всі три (`≤ 10 %`: Tenure, WarehouseToHome, …; `10–40 %`: CashbackAmount; `> 40 %`: OrderAmountHikeFromlastYear), категоріальні — обидві (`≤ 10 %`: PreferredLoginDevice, PreferedOrderCat, MaritalStatus; `> 10 %`: PreferredPaymentMode → категорія Missing)."),
    ("md", "## 5. Перевірка схеми ключів: тест на множення колонки на константу\n\nКритерій ТЗ (п. 3.4.4): якщо помножити одну числову ознаку на константу, вихід `scale_invariant`-гілки не має змінитись, а `scale_sensitive` — має."),
    ("code", "Xs_train, Xs_test = X_train.copy(), X_test.copy()\nXs_train['CashbackAmount'] *= 1000; Xs_test['CashbackAmount'] *= 1000\nfor name in ['config_lr', 'config_rf']:\n    path = str(ROOT / f'config/{name}.yaml')\n    a = DynamicPipelineBuilder(path).build().fit(X_train, y_train).transform(X_test)\n    b = DynamicPipelineBuilder(path).build().fit(Xs_train, y_train).transform(Xs_test)\n    fam = yaml.safe_load(open(path, encoding='utf-8'))['model_family']\n    print(f'{fam:16s} max|Δ| = {np.abs(np.asarray(a, float) - np.asarray(b, float)).max():.3g}')"),
    ("md", "## 6. Вихідна схема конвеєра\n\nЖодної захищеної ознаки серед вихідних імен; кількість ознак відрізняється між родинами через індикатори пропусків і спосіб кодування."),
    ("code", "names = list(ct.get_feature_names_out()); names_inv = list(ct_inv.get_feature_names_out())\nprint(len(names), 'ознак (scale_sensitive):', names)\nprint(len(names_inv), 'ознак (scale_invariant):', names_inv)\nassert not any(p in n for n in names + names_inv for p in ['CityTier', 'Gender'])"),
]

if __name__ == "__main__":
    nbf.write(nb(EDA), NB / "00_eda.ipynb")
    nbf.write(nb(QG), NB / "01_quality_gate.ipynb")
    print("ноутбуки записано")
