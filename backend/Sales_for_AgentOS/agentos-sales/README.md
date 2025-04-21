```markdown
# AgentOS ‑ Sales Service

Microserviço que orquestra o ciclo de vida de uma venda no ecossistema **AgentOS/VoulezVous**.

* **Stack:** FastAPI + Motor(MongoDB), Celery + Redis, Pydantic v2, Loguru.
* **Padrões:** Repository + Service Layer, tarefas assíncronas para integrações (Banking, Delivery, Pessoas).
* **Observabilidade:** logs estruturados, eventos Redis (`sales‑updates`), métricas Prometheus.
* **Como rodar:**

```bash
cp .env.example .env
make dev       # API em :8000 com reload
make worker    # Celery worker

Documentação completa em /docs (Swagger) quando a API estiver no ar.
```