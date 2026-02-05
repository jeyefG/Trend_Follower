# Trend Follower XAUUSD (Python + MT5) — AGENTS.md

## Objetivo
Construir un sistema trend following automático (MVP) para XAUUSD con:
- reglas mecánicas claras
- backtest realista (costos desde el inicio)
- walk-forward
- paper/shadow
- live micro con kill-switch
- escalamiento gradual

## Principios
- Simple > sofisticado. 1 estrategia core.
- Sin p-hacking: pocas iteraciones, cambios versionados (v1, v1.1, etc.).
- Costos reales SIEMPRE: spread + comisión + slippage.
- Robustez: debe sobrevivir stress test con costos 2×.
- Auditable: logging de decisiones y trades; output reproducible.
- Automatizable: diseño compatible con ejecución real en MT5.

## Estrategia (v1) — TF-DC-ATR
Timeframe: H1 (decisión al cierre, ejecución al open siguiente)
Filtro tendencia:
- EMA200
- slope24 = EMA200[t] - EMA200[t-24]
- LONG permitido si Close>EMA200 y slope24 > 0.25*ATR14
- SHORT permitido si Close<EMA200 y slope24 < -0.25*ATR14
Entrada:
- Donchian breakout N=55 por close:
  - LONG si Close[t] > DonchianHigh55[t-1] y trend LONG permitido
  - SHORT si Close[t] < DonchianLow55[t-1] y trend SHORT permitido
- Ejecutar market en open[t+1]
Stops:
- SL inicial: 2.8*ATR14 desde entry
- Trailing: Chandelier 22 con 3.0*ATR14 (update al cierre H1)
Time-stop:
- 120 velas H1: si no alcanza +1R en MFE, cerrar en open[t+1]
Filtros NO operar:
- Spread filter: spread > 0.12*ATR => no entry
- ATR min filter: ATR < 0.70 * medianATR200 => no entry
- Entry hours: 07:00–20:00 UTC (solo nuevas entradas)
Cooldown:
- 24 velas H1 post-exit sin nuevas entradas

## Backtest (event-driven)
- Sin leakage: señales solo con datos <= cierre de t
- Ejecución: next-bar open
- Stops: gatillan intrabar usando High/Low; fill a precio de stop +/- slippage
- Costos: spread + comisión + slippage modelados explícitamente
- Métricas mínimas:
  - expectancy (R neto/trade)
  - win rate, payoff, trades/semana
  - maxDD en R y en equity
  - recovery time
  - sensibilidad a costos (1× y 2×)
  - estabilidad por subperiodos (rolling)

## GO/NO-GO (automático)
GO si:
- expectancy_net >= 0.10R/trade
- maxDD <= 20R
- con costos 2× expectancy_net >= 0R
- frecuencia ~1–5 trades/semana (rango aceptable: 0.5–8)
NO-GO si falla cualquiera o requiere micro-tuning para pasar.

## Estructura repo (esperada)
- /src
  - data/mt5_client.py
  - strategy/tf_dc_atr.py
  - backtest/engine.py
  - backtest/costs.py
  - backtest/metrics.py
  - walkforward/protocol.py
  - live/runner.py (más adelante)
- /configs
  - xauusd_v1.yaml
- /scripts
  - run_backtest.py
  - run_walkforward.py
- /tests (unit tests mínimos)
- /artifacts (outputs: csv, logs)

## Iteración con Codex (regla anti-loop)
- Codex debe responder primero con un "Design Plan" breve.
- Solo 1 ronda de ajustes por entrega.
- Si el plan está razonable: GO y se implementa.
- Si hay 1-2 fallas graves: NO-GO, se corrige plan y se reintenta una sola vez.
- No se permiten 10 micro-ajustes: se prioriza avanzar.
