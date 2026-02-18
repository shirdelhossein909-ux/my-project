#ifndef __PROPSHIELD_PROP_RULES_MQH__
#define __PROPSHIELD_PROP_RULES_MQH__

#include "PropTypes.mqh"

class CPropRules
{
private:
   SPropRuleSet m_rules[];

public:
   void LoadDefaultIranTop5Strict()
   {
      ArrayResize(m_rules, 5);

      // اعداد از نسخه رایج بازار کمی سخت‌گیرانه‌تر تنظیم شده‌اند.
      m_rules[0].name              = "Sarmaye Aval";
      m_rules[0].dailyLossPct      = 4.50;
      m_rules[0].overallLossPct    = 9.00;
      m_rules[0].maxRiskPerTradePct= 1.00;
      m_rules[0].requireStopLoss   = true;

      m_rules[1].name              = "PersianFunded";
      m_rules[1].dailyLossPct      = 4.00;
      m_rules[1].overallLossPct    = 8.50;
      m_rules[1].maxRiskPerTradePct= 0.90;
      m_rules[1].requireStopLoss   = true;

      m_rules[2].name              = "Tejarat Prop";
      m_rules[2].dailyLossPct      = 3.80;
      m_rules[2].overallLossPct    = 8.00;
      m_rules[2].maxRiskPerTradePct= 0.80;
      m_rules[2].requireStopLoss   = true;

      m_rules[3].name              = "Aria Capital";
      m_rules[3].dailyLossPct      = 4.20;
      m_rules[3].overallLossPct    = 8.80;
      m_rules[3].maxRiskPerTradePct= 0.90;
      m_rules[3].requireStopLoss   = true;

      m_rules[4].name              = "Pars Profit";
      m_rules[4].dailyLossPct      = 3.50;
      m_rules[4].overallLossPct    = 7.50;
      m_rules[4].maxRiskPerTradePct= 0.75;
      m_rules[4].requireStopLoss   = true;
   }

   int Count() const
   {
      return ArraySize(m_rules);
   }

   bool GetByIndex(const int index, SPropRuleSet &outRules) const
   {
      if(index < 0 || index >= ArraySize(m_rules))
         return false;

      outRules = m_rules[index];
      return true;
   }
};

#endif
