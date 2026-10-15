#!/bin/bash
# Zero-token SLURM watcher for the laptop: every 5 minutes one short sacct call over the ControlMaster connection;
# appends one line per job state change to jobs/.events. The heartbeat reads that file instead of polling the cluster.
cd "$(dirname "$0")/.." || exit 1
STATE=jobs/.watch_state
EVENTS=jobs/.events
touch "$STATE" "$EVENTS"
echo "$(date '+%Y-%m-%dT%H:%M:%S') watcher started pid $$" >> "$EVENTS"
while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=15 explorer 'sacct -u shehral.m -S now-2days -n -P -X -o jobid,jobname,state,elapsed,exitcode 2>/dev/null' 2>/dev/null) || { sleep 300; continue; }
  while IFS='|' read -r jid name st el ec; do
    [ -z "$jid" ] && continue
    # fixed-string lookups: array ids such as 10928445_[8-10%4] are not regexes (a bracket range broke grep and wiped the state)
    prev=$(awk -F'|' -v j="$jid" '$1 == j {print $2; exit}' "$STATE")
    if [ "$prev" != "$st" ]; then
      printf '%s %s %s %s elapsed=%s exit=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$jid" "$name" "$st" "$el" "$ec" >> "$EVENTS"
      awk -F'|' -v j="$jid" '$1 != j' "$STATE" > "$STATE.tmp" 2>/dev/null; echo "$jid|$st" >> "$STATE.tmp"; mv "$STATE.tmp" "$STATE"
    fi
  done <<< "$out"
  sleep 300
done
