# Live Trader V1 Safe

Diese Version macht nur einen Sicherheitsumbau.

## Änderungen
- Der Live-Trader läuft nicht mehr unendlich.
- Maximal 28 Scans.
- 15 Sekunden zwischen den Scans.
- Laufzeit des Traders ca. 7 Minuten.
- Abschluss-Cleanup vor dem Ende.
- GitHub Actions hat `concurrency`, damit keine zwei Live-Trader gleichzeitig laufen.
- GitHub Actions hat ein 9-Minuten-Timeout.

## Bewusst NICHT geändert
- Börsen
- Symbole
- Arbitrage-Logik
- Mindestspread 0,30 %
- Ordergröße 10 USDT

## Nächste Phase
Bayesian Edge + Fractional Kelly wird erst nach erfolgreichem Safe-Run eingebaut.
