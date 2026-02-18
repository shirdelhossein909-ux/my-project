#ifndef __PROPSHIELD_RISK_ENGINE_MQH__
#define __PROPSHIELD_RISK_ENGINE_MQH__

#include <Trade/PositionInfo.mqh>
#include "PropTypes.mqh"

class CRiskEngine
{
private:
   SPropRuleSet m_rules;
   bool         m_hasRules;
   bool         m_locked;
   string       m_lockReason;
   double       m_initialBalance;
   double       m_dayStartEquity;
   int          m_dayOfYear;

public:
   void Init()
   {
      m_hasRules       = false;
      m_locked         = false;
      m_lockReason     = "";
      m_initialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      m_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);

      MqlDateTime now;
      TimeToStruct(TimeCurrent(), now);
      m_dayOfYear = now.day_of_year;
   }

   void SetRules(const SPropRuleSet &rules)
   {
      m_rules = rules;
      m_hasRules = true;
   }

   bool IsLocked() const { return m_locked; }
   string LockReason() const { return m_lockReason; }
   bool HasRules() const { return m_hasRules; }

   void LockTrading(const string reason)
   {
      m_locked = true;
      m_lockReason = reason;
   }

   void RefreshDayAnchor()
   {
      MqlDateTime now;
      TimeToStruct(TimeCurrent(), now);
      if(now.day_of_year != m_dayOfYear)
      {
         m_dayOfYear = now.day_of_year;
         m_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      }
   }

   double DailyDrawdownPct() const
   {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      if(m_dayStartEquity <= 0.0)
         return 0.0;
      return MathMax(0.0, (m_dayStartEquity - eq) / m_dayStartEquity * 100.0);
   }

   double OverallDrawdownPct() const
   {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      if(m_initialBalance <= 0.0)
         return 0.0;
      return MathMax(0.0, (m_initialBalance - eq) / m_initialBalance * 100.0);
   }

   bool CheckGlobalLimits(string &viol)
   {
      viol = "";
      if(!m_hasRules)
         return true;

      const double ddDay = DailyDrawdownPct();
      const double ddAll = OverallDrawdownPct();

      if(ddDay >= m_rules.dailyLossPct)
      {
         viol = StringFormat("Daily DD violated: %.2f%% >= %.2f%%", ddDay, m_rules.dailyLossPct);
         return false;
      }

      if(ddAll >= m_rules.overallLossPct)
      {
         viol = StringFormat("Overall DD violated: %.2f%% >= %.2f%%", ddAll, m_rules.overallLossPct);
         return false;
      }

      return true;
   }

   bool ValidatePositionRisk(const string symbol,
                             const ENUM_POSITION_TYPE posType,
                             const double volume,
                             const double entryPrice,
                             const double stopLoss,
                             string &viol) const
   {
      viol = "";
      if(!m_hasRules)
         return true;

      if(m_rules.requireStopLoss && (stopLoss <= 0.0))
      {
         viol = "Missing stop-loss";
         return false;
      }

      if(stopLoss <= 0.0)
         return true;

      const double tickSize  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
      const double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
      if(tickSize <= 0 || tickValue <= 0)
      {
         viol = "Cannot calculate risk (tick data invalid)";
         return false;
      }

      const double stopDist = MathAbs(entryPrice - stopLoss);
      if(stopDist <= 0.0)
      {
         viol = "Invalid stop-loss distance";
         return false;
      }

      const double riskMoney = (stopDist / tickSize) * tickValue * volume;
      const double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(equity <= 0.0)
      {
         viol = "Invalid equity";
         return false;
      }

      const double riskPct = riskMoney / equity * 100.0;
      if(riskPct > m_rules.maxRiskPerTradePct)
      {
         viol = StringFormat("Risk per trade exceeded: %.2f%% > %.2f%%", riskPct, m_rules.maxRiskPerTradePct);
         return false;
      }

      return true;
   }

   bool Rules(SPropRuleSet &outRules) const
   {
      if(!m_hasRules)
         return false;
      outRules = m_rules;
      return true;
   }
};

#endif
