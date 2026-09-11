# Streamlit

> streamlit run web/app.py


# API

> uvicorn api.server:app --reload


Jogar logs do projeto na AWS

> aws s3 cp observability/logs/dev/ s3://chatbot-agent-genai-logs/AWSLogs/251675404411/BedrockModelInvocationLogs/us-east-1/2026/04/08/20/ --recursive


Tabela do Athena para consultar os logs

```sql
CREATE EXTERNAL TABLE bedrock_logs_raw (
  timestamp string,
  accountId string,
  region string,
  requestId string,
  operation string,
  modelId string,
  input struct<
    inputTokenCount:int
  >,
  output struct<
    outputBodyJson:struct<
      usage:struct<
        inputTokens:int,
        outputTokens:int,
        totalTokens:int
      >,
      metrics:struct<
        latencyMs:int
      >
    >
  >
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://chatbot-agent-genai-logs/AWSLogs/251675404411/BedrockModelInvocationLogs/';
```

Logs do app

```sql
CREATE EXTERNAL TABLE app_logs_raw (
  timestamp string,
  environment string,
  application string,
  agent_name string,
  agent_version string,
  request_id string,
  route string,
  tool_name string,
  model_id string,
  latency_ms double,
  prompt_tokens int,
  completion_tokens int,
  total_tokens int,
  estimated_cost_usd double,
  status string,
  error_type string,
  error_message string,
  user_text string,
  answer_preview string
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://chatbot-agent-genai-logs/AWSLogs/251675404411/app/';
```