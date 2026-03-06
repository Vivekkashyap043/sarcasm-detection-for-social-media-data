# Azure Deployment Steps (Frontend -> Keyword API -> Sarcasm Results)

This guide deploys your project so a frontend can send keywords and get JSON sarcasm predictions.

## 1) What was added in code

- API server: `api_server.py`
- Endpoint: `POST /predict/keywords`
- Health check: `GET /health`
- Social pipeline refactor for API reuse: `collect_pipeline_results(...)` in `social_media_pipeline.py`
- Training/testing artifact storage improved:
  - `results/training_history.json`
  - `results/training_history.csv`
  - `results/training_accuracy_curve.png`
  - `results/training_loss_curve.png`
  - evaluation files already produced by evaluator:
    - `results/evaluation_results.json`
    - `results/evaluation_results.csv`
    - `results/evaluation_report.txt`

## 2) Train and store all metrics/plots locally first

Run:

```bash
python train.py
```

This now stores per-epoch train/val/test metrics and plots in `results/`.

Run test evaluation:

```bash
python test.py test --model models/<best_model>.pth
```

This stores confusion matrix, accuracy, precision, recall, f1-score and detailed report files.

## 3) API request format for frontend

Endpoint:

```text
POST /predict/keywords
Content-Type: application/json
```

Body example:

```json
{
  "keywords": ["Donald Trump", "JD Vance"],
  "subreddits": ["news", "politics"],
  "posts_per_subreddit": 10,
  "comments_per_post": 5,
  "detailed_explanations": false,
  "output_path": "results/reddit_multimodal_results.json"
}
```

Response includes:

- `total_results`
- `output_path`
- `results` (full prediction payload)

## 4) Local API run command

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl http://localhost:8000/health
```

## 5) Deploy to Azure App Service (recommended quick path)

## 5.1 Prerequisites

- Azure CLI installed and logged in: `az login`
- Resource group created (or let commands create one)
- Python runtime 3.11+

## 5.2 Create resources

```bash
az group create --name sarcasm-rg --location eastus
az appservice plan create --name sarcasm-plan --resource-group sarcasm-rg --sku B1 --is-linux
az webapp create --resource-group sarcasm-rg --plan sarcasm-plan --name <unique-app-name> --runtime "PYTHON|3.11"
```

## 5.3 Configure startup command

```bash
az webapp config set --resource-group sarcasm-rg --name <unique-app-name> --startup-file "uvicorn api_server:app --host 0.0.0.0 --port 8000"
az webapp config appsettings set --resource-group sarcasm-rg --name <unique-app-name> --settings WEBSITES_PORT=8000 SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

## 5.4 Deploy code

Option A (zip deploy):

```bash
az webapp deployment source config-zip --resource-group sarcasm-rg --name <unique-app-name> --src <project-zip-file>
```

Option B (GitHub Actions from Azure portal) is also valid.

## 5.5 Verify deployment

```bash
curl https://<unique-app-name>.azurewebsites.net/health
```

## 6) Frontend integration notes

Use your app URL:

```text
https://<unique-app-name>.azurewebsites.net/predict/keywords
```

Typical frontend fetch:

```javascript
const res = await fetch("https://<app>.azurewebsites.net/predict/keywords", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ keywords: ["Donald Trump"], posts_per_subreddit: 5, comments_per_post: 3 })
});
const data = await res.json();
```

## 7) Production recommendations

1. Add CORS policy in API for your frontend domain.
2. Add request timeout and rate limiting.
3. Use Azure Blob Storage if result files must persist across instance restarts.
4. Use Azure Key Vault for secrets if authenticated APIs are introduced later.
5. Consider Azure Container Apps if you need stronger scaling control.

## 8) Minimal cleanup done safely

Only obvious runtime cache folders should be removed automatically:

- `__pycache__/`
- `src/__pycache__/`

For deleting project scripts/files, do a reviewed cleanup list first to avoid removing needed code.
