#!/usr/bin/env bash
set -u
END=$(date -d "tomorrow 09:00:00" +%s)
START=$(date +%s)

while [ $(date +%s) -lt $END ]; do
  cd /e/Myst_Studio || exit 1
  echo "=== AUTONOMOUS PASS $(date -Is) ==="
  python -m py_compile /e/Myst_Studio/11_Myst_AI/myst_ai_server.py /e/Myst_Studio/build_mrai_public.py
  python /e/Myst_Studio/build_mrai_public.py
  timeout 8 python /e/Myst_Studio/11_Myst_AI/myst_ai_server.py > /e/Myst_Studio/srv.log 2>&1 &
  SRVPID=$!
  sleep 4
  STATS=$(curl -s http://127.0.0.1:5005/api/stats)
  CHAT=$(curl -s -X POST http://127.0.0.1:5005/api/chat -H "Content-Type: application/json" -d '{"message":"hello"}')
  kill $SRVPID 2>/dev/null
  wait $SRVPID 2>/dev/null
  echo "STATS: $STATS"
  echo "CHAT: $CHAT"
  if echo "$CHAT" | grep -q '"status": "ok"'; then
    echo "AUTONOMOUS_CHECK=PASS"
    powershell -NoProfile -ExecutionPolicy Bypass -File /e/Myst_Studio/download_models.ps1
  else
    echo "AUTONOMOUS_CHECK=FAIL"
    exit 1
  fi
  sleep 60
done
echo "AUTONOMOUS_WINDOW_DONE"
