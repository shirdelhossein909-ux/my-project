#ifndef __PROPSHIELD_UI_PANEL_MQH__
#define __PROPSHIELD_UI_PANEL_MQH__

#include "PropTypes.mqh"

class CUIPanel
{
private:
   long   m_chartId;
   string m_prefix;
   bool   m_fullscreen;

   int    m_panelX;
   int    m_panelY;
   int    m_panelW;
   int    m_panelH;

   string Name(const string id) const { return m_prefix + id; }

   void ApplyCorner(const string objName) const
   {
      ObjectSetInteger(m_chartId, objName, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
      ObjectSetInteger(m_chartId, objName, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
   }

   void PlaceLabel(const string id, const int x, const int y)
   {
      const string name = Name(id);
      ObjectSetInteger(m_chartId, name, OBJPROP_XDISTANCE, m_panelX + x);
      ObjectSetInteger(m_chartId, name, OBJPROP_YDISTANCE, m_panelY + y);
   }

   void SetButtonStyle(const string name,
                       const int x,
                       const int y,
                       const int w,
                       const int h,
                       const string text,
                       const color fg,
                       const color bg,
                       const color border,
                       const bool shadow)
   {
      if(ObjectFind(m_chartId, name) < 0)
         ObjectCreate(m_chartId, name, OBJ_BUTTON, 0, 0, 0);

      ApplyCorner(name);
      ObjectSetInteger(m_chartId, name, OBJPROP_XDISTANCE, x);
      ObjectSetInteger(m_chartId, name, OBJPROP_YDISTANCE, y);
      ObjectSetInteger(m_chartId, name, OBJPROP_XSIZE, w);
      ObjectSetInteger(m_chartId, name, OBJPROP_YSIZE, h);
      ObjectSetString(m_chartId, name, OBJPROP_TEXT, text);
      ObjectSetInteger(m_chartId, name, OBJPROP_COLOR, fg);
      ObjectSetInteger(m_chartId, name, OBJPROP_BGCOLOR, bg);
      ObjectSetInteger(m_chartId, name, OBJPROP_BORDER_COLOR, border);
      ObjectSetInteger(m_chartId, name, OBJPROP_FONTSIZE, 10);
      ObjectSetInteger(m_chartId, name, OBJPROP_STATE, false);
      ObjectSetString(m_chartId, name, OBJPROP_FONT, "Segoe UI");
      ObjectSetInteger(m_chartId, name, OBJPROP_HIDDEN, false);
      ObjectSetInteger(m_chartId, name, OBJPROP_BACK, false);

      // pseudo shadow via thin border depth
      if(shadow)
         ObjectSetInteger(m_chartId, name, OBJPROP_ZORDER, 5);
   }

public:
   void Init(const long chartId, const string prefix = "PropShield_")
   {
      m_chartId = chartId;
      m_prefix = prefix;
      m_fullscreen = false;
      m_panelX = 14;
      m_panelY = 18;
      m_panelW = 470;
      m_panelH = 340;
   }

   bool IsFullscreen() const { return m_fullscreen; }

   void CreateLabel(const string id, const int x, const int y, const string text, const color c)
   {
      const string name = Name(id);
      if(ObjectFind(m_chartId, name) < 0)
         ObjectCreate(m_chartId, name, OBJ_LABEL, 0, 0, 0);

      ApplyCorner(name);
      ObjectSetInteger(m_chartId, name, OBJPROP_XDISTANCE, m_panelX + x);
      ObjectSetInteger(m_chartId, name, OBJPROP_YDISTANCE, m_panelY + y);
      ObjectSetString(m_chartId, name, OBJPROP_TEXT, text);
      ObjectSetInteger(m_chartId, name, OBJPROP_COLOR, c);
      ObjectSetInteger(m_chartId, name, OBJPROP_FONTSIZE, 11);
      ObjectSetString(m_chartId, name, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(m_chartId, name, OBJPROP_BACK, false);
   }

   void CreateBasePanel()
   {
      const string shadow = Name("SHADOW");
      ObjectCreate(m_chartId, shadow, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ApplyCorner(shadow);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_XDISTANCE, m_panelX + 4);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_YDISTANCE, m_panelY + 4);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_XSIZE, m_panelW);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_YSIZE, m_panelH);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_BGCOLOR, C'12,12,18');
      ObjectSetInteger(m_chartId, shadow, OBJPROP_COLOR, C'12,12,18');
      ObjectSetInteger(m_chartId, shadow, OBJPROP_BORDER_TYPE, BORDER_FLAT);

      const string bg = Name("BG");
      ObjectCreate(m_chartId, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ApplyCorner(bg);
      ObjectSetInteger(m_chartId, bg, OBJPROP_XDISTANCE, m_panelX);
      ObjectSetInteger(m_chartId, bg, OBJPROP_YDISTANCE, m_panelY);
      ObjectSetInteger(m_chartId, bg, OBJPROP_XSIZE, m_panelW);
      ObjectSetInteger(m_chartId, bg, OBJPROP_YSIZE, m_panelH);
      ObjectSetInteger(m_chartId, bg, OBJPROP_BGCOLOR, C'8,10,16');
      ObjectSetInteger(m_chartId, bg, OBJPROP_COLOR, C'42,58,79');
      ObjectSetInteger(m_chartId, bg, OBJPROP_WIDTH, 1);
      ObjectSetInteger(m_chartId, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(m_chartId, bg, OBJPROP_BACK, false);

      const string graph = Name("GRAPH_BG");
      ObjectCreate(m_chartId, graph, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ApplyCorner(graph);
      ObjectSetInteger(m_chartId, graph, OBJPROP_XDISTANCE, m_panelX + 18);
      ObjectSetInteger(m_chartId, graph, OBJPROP_YDISTANCE, m_panelY + 190);
      ObjectSetInteger(m_chartId, graph, OBJPROP_XSIZE, m_panelW - 36);
      ObjectSetInteger(m_chartId, graph, OBJPROP_YSIZE, 130);
      ObjectSetInteger(m_chartId, graph, OBJPROP_BGCOLOR, C'13,18,28');
      ObjectSetInteger(m_chartId, graph, OBJPROP_COLOR, C'36,61,83');
      ObjectSetInteger(m_chartId, graph, OBJPROP_BORDER_TYPE, BORDER_FLAT);

      CreateLabel("TITLE", 20, 18, "PropShield PRO UI", C'119,255,171');
      CreateLabel("SUB", 20, 40, "Smart Prop Control Panel", C'145,174,212');
      CreateLabel("PROP", 20, 72, "Prop: Not Selected", clrWhite);
      CreateLabel("STATE", 20, 98, "State: Waiting Selection", C'255,231,106');
      CreateLabel("DD_DAY", 20, 124, "Daily DD: 0.00%", clrWhite);
      CreateLabel("DD_ALL", 20, 146, "Overall DD: 0.00%", clrWhite);
      CreateLabel("RISK", 20, 168, "Max Risk/Trade: -", clrWhite);
      CreateLabel("LOCK", 20, 190, "Lock: No", C'77,235,255');
      CreateLabel("GRAPH", 34, 215, "Equity graph available in fullscreen", C'125,177,229');

      SetButtonStyle(Name("FULL_BTN"), m_panelX + m_panelW - 152, m_panelY + 14, 130, 28,
                     "⛶ Fullscreen", clrWhite, C'32,52,78', C'80,118,166', true);
   }

   void CreatePropButton(const int index, const string caption)
   {
      const int w = 196;
      const int h = 32;
      const int gapX = 20;
      const int gapY = 14;
      const int baseX = m_panelX + 20;
      const int baseY = m_panelY + 230;
      const int col = index % 2;
      const int row = index / 2;
      const int x = baseX + col * (w + gapX);
      const int y = baseY + row * (h + gapY);

      const string shadow = Name(StringFormat("PROP_BTN_SHADOW_%d", index));
      if(ObjectFind(m_chartId, shadow) < 0)
         ObjectCreate(m_chartId, shadow, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ApplyCorner(shadow);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_XDISTANCE, x + 2);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_YDISTANCE, y + 2);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_XSIZE, w);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_YSIZE, h);
      ObjectSetInteger(m_chartId, shadow, OBJPROP_BGCOLOR, C'17,24,35');
      ObjectSetInteger(m_chartId, shadow, OBJPROP_COLOR, C'17,24,35');
      ObjectSetInteger(m_chartId, shadow, OBJPROP_BORDER_TYPE, BORDER_FLAT);

      SetButtonStyle(Name(StringFormat("PROP_BTN_%d", index)),
                     x, y, w, h,
                     caption,
                     clrWhite,
                     C'36,56,84',
                     C'89,132,188',
                     true);
   }

   bool IsPropButton(const string objName, int &indexOut) const
   {
      for(int i = 0; i < 10; i++)
      {
         if(objName == Name(StringFormat("PROP_BTN_%d", i)))
         {
            indexOut = i;
            return true;
         }
      }
      return false;
   }

   bool IsFullscreenButton(const string objName) const
   {
      return objName == Name("FULL_BTN");
   }

   void ToggleFullscreen()
   {
      m_fullscreen = !m_fullscreen;
      if(m_fullscreen)
      {
         m_panelX = 8;
         m_panelY = 8;
         m_panelW = 760;
         m_panelH = 500;
      }
      else
      {
         m_panelX = 14;
         m_panelY = 18;
         m_panelW = 470;
         m_panelH = 340;
      }

      // relocate layers
      ObjectSetInteger(m_chartId, Name("SHADOW"), OBJPROP_XDISTANCE, m_panelX + 4);
      ObjectSetInteger(m_chartId, Name("SHADOW"), OBJPROP_YDISTANCE, m_panelY + 4);
      ObjectSetInteger(m_chartId, Name("SHADOW"), OBJPROP_XSIZE, m_panelW);
      ObjectSetInteger(m_chartId, Name("SHADOW"), OBJPROP_YSIZE, m_panelH);

      ObjectSetInteger(m_chartId, Name("BG"), OBJPROP_XDISTANCE, m_panelX);
      ObjectSetInteger(m_chartId, Name("BG"), OBJPROP_YDISTANCE, m_panelY);
      ObjectSetInteger(m_chartId, Name("BG"), OBJPROP_XSIZE, m_panelW);
      ObjectSetInteger(m_chartId, Name("BG"), OBJPROP_YSIZE, m_panelH);

      ObjectSetInteger(m_chartId, Name("GRAPH_BG"), OBJPROP_XDISTANCE, m_panelX + 18);
      ObjectSetInteger(m_chartId, Name("GRAPH_BG"), OBJPROP_YDISTANCE, m_panelY + 190);
      ObjectSetInteger(m_chartId, Name("GRAPH_BG"), OBJPROP_XSIZE, m_panelW - 36);
      ObjectSetInteger(m_chartId, Name("GRAPH_BG"), OBJPROP_YSIZE, m_fullscreen ? 280 : 130);

      PlaceLabel("TITLE", 20, 18);
      PlaceLabel("SUB", 20, 40);
      PlaceLabel("PROP", 20, 72);
      PlaceLabel("STATE", 20, 98);
      PlaceLabel("DD_DAY", 20, 124);
      PlaceLabel("DD_ALL", 20, 146);
      PlaceLabel("RISK", 20, 168);
      PlaceLabel("LOCK", 20, 190);
      PlaceLabel("GRAPH", 34, 215);

      SetButtonStyle(Name("FULL_BTN"), m_panelX + m_panelW - 152, m_panelY + 14, 130, 28,
                     m_fullscreen ? "🗗 Minimize" : "⛶ Fullscreen",
                     clrWhite, C'32,52,78', C'80,118,166', true);
   }

   void HidePropButtons(const int count)
   {
      for(int i = 0; i < count; i++)
      {
         ObjectDelete(m_chartId, Name(StringFormat("PROP_BTN_%d", i)));
         ObjectDelete(m_chartId, Name(StringFormat("PROP_BTN_SHADOW_%d", i)));
      }
   }

   void HandleMouseMove(const int mouseX, const int mouseY, const int count)
   {
      for(int i = 0; i < count; i++)
      {
         const string btn = Name(StringFormat("PROP_BTN_%d", i));
         if(ObjectFind(m_chartId, btn) < 0)
            continue;

         int x = (int)ObjectGetInteger(m_chartId, btn, OBJPROP_XDISTANCE);
         int y = (int)ObjectGetInteger(m_chartId, btn, OBJPROP_YDISTANCE);
         int w = (int)ObjectGetInteger(m_chartId, btn, OBJPROP_XSIZE);
         int h = (int)ObjectGetInteger(m_chartId, btn, OBJPROP_YSIZE);

         bool hover = (mouseX >= x && mouseX <= (x + w) && mouseY >= y && mouseY <= (y + h));
         ObjectSetInteger(m_chartId, btn, OBJPROP_XSIZE, hover ? 202 : 196);
         ObjectSetInteger(m_chartId, btn, OBJPROP_YSIZE, hover ? 34 : 32);
         ObjectSetInteger(m_chartId, btn, OBJPROP_BGCOLOR, hover ? C'51,84,122' : C'36,56,84');
         ObjectSetInteger(m_chartId, btn, OBJPROP_BORDER_COLOR, hover ? C'131,188,255' : C'89,132,188');
      }

      const string fullBtn = Name("FULL_BTN");
      int fx = (int)ObjectGetInteger(m_chartId, fullBtn, OBJPROP_XDISTANCE);
      int fy = (int)ObjectGetInteger(m_chartId, fullBtn, OBJPROP_YDISTANCE);
      int fw = (int)ObjectGetInteger(m_chartId, fullBtn, OBJPROP_XSIZE);
      int fh = (int)ObjectGetInteger(m_chartId, fullBtn, OBJPROP_YSIZE);
      bool fhover = (mouseX >= fx && mouseX <= (fx + fw) && mouseY >= fy && mouseY <= (fy + fh));
      ObjectSetInteger(m_chartId, fullBtn, OBJPROP_BGCOLOR, fhover ? C'43,71,105' : C'32,52,78');
   }

   void UpdateStatus(const string propName,
                     const string state,
                     const double dailyDd,
                     const double overallDd,
                     const double maxRisk,
                     const bool locked,
                     const string lockReason,
                     const string equityGraph)
   {
      ObjectSetString(m_chartId, Name("PROP"), OBJPROP_TEXT, "Prop: " + propName);
      ObjectSetString(m_chartId, Name("STATE"), OBJPROP_TEXT, "State: " + state);
      ObjectSetString(m_chartId, Name("DD_DAY"), OBJPROP_TEXT, StringFormat("Daily DD: %.2f%%", dailyDd));
      ObjectSetString(m_chartId, Name("DD_ALL"), OBJPROP_TEXT, StringFormat("Overall DD: %.2f%%", overallDd));

      ObjectSetString(m_chartId, Name("RISK"), OBJPROP_TEXT,
                      (maxRisk > 0.0) ? StringFormat("Max Risk/Trade: %.2f%%", maxRisk) : "Max Risk/Trade: -");

      if(locked)
      {
         ObjectSetString(m_chartId, Name("LOCK"), OBJPROP_TEXT, "Lock: YES | " + lockReason);
         ObjectSetInteger(m_chartId, Name("LOCK"), OBJPROP_COLOR, clrTomato);
      }
      else
      {
         ObjectSetString(m_chartId, Name("LOCK"), OBJPROP_TEXT, "Lock: No");
         ObjectSetInteger(m_chartId, Name("LOCK"), OBJPROP_COLOR, C'77,235,255');
      }

      if(m_fullscreen)
         ObjectSetString(m_chartId, Name("GRAPH"), OBJPROP_TEXT, "Equity Growth\n" + equityGraph);
      else
         ObjectSetString(m_chartId, Name("GRAPH"), OBJPROP_TEXT, "Click Fullscreen for Equity Growth chart");
   }
};

#endif
