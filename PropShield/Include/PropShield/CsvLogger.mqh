#ifndef __PROPSHIELD_CSV_LOGGER_MQH__
#define __PROPSHIELD_CSV_LOGGER_MQH__

class CCsvLogger
{
private:
   string m_fileName;

   string TodayTag() const
   {
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      return StringFormat("%04d-%02d-%02d", dt.year, dt.mon, dt.day);
   }

public:
   void Init(const string prefix = "PropShield")
   {
      m_fileName = StringFormat("%s_%s.csv", prefix, TodayTag());
      int h = FileOpen(m_fileName, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI, ';');
      if(h == INVALID_HANDLE)
         return;

      if(FileSize(h) == 0)
         FileWrite(h, "time", "event", "prop", "equity", "balance", "daily_dd_pct", "overall_dd_pct", "details");

      FileClose(h);
   }

   void Log(const string eventName,
            const string propName,
            const double equity,
            const double balance,
            const double dailyDdPct,
            const double overallDdPct,
            const string details)
   {
      if(m_fileName == "")
         Init();

      int h = FileOpen(m_fileName, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI, ';');
      if(h == INVALID_HANDLE)
         return;

      FileSeek(h, 0, SEEK_END);
      FileWrite(h,
                TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
                eventName,
                propName,
                DoubleToString(equity, 2),
                DoubleToString(balance, 2),
                DoubleToString(dailyDdPct, 2),
                DoubleToString(overallDdPct, 2),
                details);

      FileClose(h);
   }
};

#endif
