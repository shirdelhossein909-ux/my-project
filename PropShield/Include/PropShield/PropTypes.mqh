#ifndef __PROPSHIELD_PROP_TYPES_MQH__
#define __PROPSHIELD_PROP_TYPES_MQH__

struct SPropRuleSet
{
   string name;
   double dailyLossPct;      // Max daily loss (% of day-start equity)
   double overallLossPct;    // Max overall loss (% of initial balance)
   double maxRiskPerTradePct;
   bool   requireStopLoss;
};

#endif
