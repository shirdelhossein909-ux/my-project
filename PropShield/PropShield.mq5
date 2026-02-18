#property strict
#property description "PropShield MVP - Risk Lock Expert for MT5"
#property version   "0.20"

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>
#include "Include/PropShield/PropTypes.mqh"
#include "Include/PropShield/PropRules.mqh"
#include "Include/PropShield/RiskEngine.mqh"
#include "Include/PropShield/CsvLogger.mqh"
#include "Include/PropShield/UIPanel.mqh"

CTrade         g_trade;
CPositionInfo  g_pos;
CPropRules     g_ruleBook;
CRiskEngine    g_risk;
CCsvLogger     g_log;
CUIPanel       g_ui;

bool         g_propSelected = false;
bool         g_settingsLocked = false;
string       g_selectedProp = "Not Selected";
SPropRuleSet g_activeRule;

datetime     g_lastScan = 0;
int          g_initialDeals = 0;
int          g_initialPos = 0;

void LockEverything(const string reason)
{
   g_risk.LockTrading(reason);
   g_log.Log("LOCK", g_selectedProp,
             AccountInfoDouble(ACCOUNT_EQUITY),
             AccountInfoDouble(ACCOUNT_BALANCE),
             g_risk.DailyDrawdownPct(),
             g_risk.OverallDrawdownPct(),
             reason);
}

bool IsFirstTradeDone()
{
   if(PositionsTotal() > g_initialPos)
      return true;

   if(HistoryDealsTotal() > g_initialDeals)
      return true;

   return false;
}

string BuildEquityGraphText(const int maxPoints = 42)
{
   const string shades = " .:-=+*#%@";
   const int shadeN = StringLen(shades);

   int total = HistoryDealsTotal();
   if(total <= 0)
      return "No history data yet";

   int start = MathMax(0, total - maxPoints);
   double curve[];
   ArrayResize(curve, total - start);

   double eq = 0.0;
   int k = 0;
   for(int i = start; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;

      long entryType = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entryType != DEAL_ENTRY_OUT)
         continue;

      double p = HistoryDealGetDouble(ticket, DEAL_PROFIT)
               + HistoryDealGetDouble(ticket, DEAL_SWAP)
               + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      eq += p;
      curve[k] = eq;
      k++;
   }

   if(k <= 1)
      return "Need at least 2 closed trades";

   ArrayResize(curve, k);

   double minV = curve[0], maxV = curve[0];
   for(int j = 1; j < k; j++)
   {
      if(curve[j] < minV) minV = curve[j];
      if(curve[j] > maxV) maxV = curve[j];
   }

   double span = maxV - minV;
   if(span <= 0.0000001)
      span = 1.0;

   string out = "";
   for(int n = 0; n < k; n++)
   {
      double ratio = (curve[n] - minV) / span;
      int idx = (int)MathRound(ratio * (shadeN - 1));
      idx = MathMax(0, MathMin(shadeN - 1, idx));
      out += StringSubstr(shades, idx, 1);
   }

   return out + "\nPnL range: " + DoubleToString(minV, 2) + " .. " + DoubleToString(maxV, 2);
}

void CloseAllPositions(const string reason)
{
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      if(!g_pos.SelectByIndex(i))
         continue;

      const string sym = g_pos.Symbol();
      if(!g_trade.PositionClose(sym))
      {
         Print("PropShield: failed to close position ", sym, " reason=", reason);
      }
   }
}

void EnforcePositionRules()
{
   if(!g_propSelected || g_risk.IsLocked())
      return;

   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      if(!g_pos.SelectByIndex(i))
         continue;

      const string sym = g_pos.Symbol();
      const double volume = g_pos.Volume();
      const double entry = g_pos.PriceOpen();
      const double sl = g_pos.StopLoss();
      const ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)g_pos.PositionType();

      string violation;
      if(!g_risk.ValidatePositionRisk(sym, ptype, volume, entry, sl, violation))
      {
         g_trade.PositionClose(sym);
         LockEverything("Position violation: " + violation);
         return;
      }
   }
}

int OnInit()
{
   g_ruleBook.LoadDefaultIranTop5Strict();
   g_risk.Init();
   g_log.Init("PropShield");

   g_initialDeals = HistoryDealsTotal();
   g_initialPos = PositionsTotal();

   ChartSetInteger(ChartID(), CHART_EVENT_MOUSE_MOVE, true);

   g_ui.Init(ChartID());
   g_ui.CreateBasePanel();

   for(int i = 0; i < g_ruleBook.Count(); i++)
   {
      SPropRuleSet r;
      if(g_ruleBook.GetByIndex(i, r))
         g_ui.CreatePropButton(i, r.name);
   }

   g_log.Log("INIT", g_selectedProp,
             AccountInfoDouble(ACCOUNT_EQUITY),
             AccountInfoDouble(ACCOUNT_BALANCE),
             0.0, 0.0,
             "EA initialized");

   EventSetTimer(1);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   g_log.Log("DEINIT", g_selectedProp,
             AccountInfoDouble(ACCOUNT_EQUITY),
             AccountInfoDouble(ACCOUNT_BALANCE),
             g_risk.DailyDrawdownPct(),
             g_risk.OverallDrawdownPct(),
             StringFormat("reason=%d", reason));
}

void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   if(id == CHARTEVENT_MOUSE_MOVE)
   {
      if(!g_propSelected && !g_settingsLocked)
         g_ui.HandleMouseMove((int)lparam, (int)dparam, g_ruleBook.Count());
      return;
   }

   if(id != CHARTEVENT_OBJECT_CLICK)
      return;

   if(g_ui.IsFullscreenButton(sparam))
   {
      g_ui.ToggleFullscreen();
      return;
   }

   if(g_settingsLocked)
      return;

   int idx = -1;
   if(!g_ui.IsPropButton(sparam, idx))
      return;

   SPropRuleSet r;
   if(!g_ruleBook.GetByIndex(idx, r))
      return;

   g_activeRule = r;
   g_propSelected = true;
   g_selectedProp = r.name;
   g_risk.SetRules(r);
   g_ui.HidePropButtons(g_ruleBook.Count());

   g_log.Log("PROP_SELECTED", g_selectedProp,
             AccountInfoDouble(ACCOUNT_EQUITY),
             AccountInfoDouble(ACCOUNT_BALANCE),
             g_risk.DailyDrawdownPct(),
             g_risk.OverallDrawdownPct(),
             StringFormat("daily=%.2f overall=%.2f risk=%.2f", r.dailyLossPct, r.overallLossPct, r.maxRiskPerTradePct));
}

void OnTimer()
{
   g_risk.RefreshDayAnchor();

   if(g_propSelected && !g_settingsLocked && IsFirstTradeDone())
   {
      g_settingsLocked = true;
      g_log.Log("SETTINGS_LOCKED", g_selectedProp,
                AccountInfoDouble(ACCOUNT_EQUITY),
                AccountInfoDouble(ACCOUNT_BALANCE),
                g_risk.DailyDrawdownPct(),
                g_risk.OverallDrawdownPct(),
                "First trade detected");
   }

   if(g_propSelected && !g_risk.IsLocked())
   {
      string viol;
      if(!g_risk.CheckGlobalLimits(viol))
      {
         CloseAllPositions(viol);
         LockEverything(viol);
      }
   }

   if(g_propSelected)
      EnforcePositionRules();

   g_ui.UpdateStatus(g_selectedProp,
                     g_propSelected ? "Monitoring" : "Waiting Selection",
                     g_risk.DailyDrawdownPct(),
                     g_risk.OverallDrawdownPct(),
                     g_propSelected ? g_activeRule.maxRiskPerTradePct : 0.0,
                     g_risk.IsLocked(),
                     g_risk.LockReason(),
                     BuildEquityGraphText());

   if(TimeCurrent() - g_lastScan >= 30)
   {
      g_lastScan = TimeCurrent();
      g_log.Log("HEARTBEAT", g_selectedProp,
                AccountInfoDouble(ACCOUNT_EQUITY),
                AccountInfoDouble(ACCOUNT_BALANCE),
                g_risk.DailyDrawdownPct(),
                g_risk.OverallDrawdownPct(),
                g_risk.IsLocked() ? g_risk.LockReason() : "OK");
   }
}

void OnTick()
{
   // Core risk logic is timer-driven for stable enforcement.
}
