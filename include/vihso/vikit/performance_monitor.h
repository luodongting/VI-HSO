#pragma once
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include "vihso/vikit/timer.h"
namespace hso
{
struct LogItem
{
  double data;
  bool   set;
};
class PerformanceMonitor
{
public:
  PerformanceMonitor();
  ~PerformanceMonitor();
  void init(const std::string& trace_name, const std::string& trace_dir);
  void addTimer(const std::string& name);
  void addLog(const std::string& name);
  void writeToFile();
  void startTimer(const std::string& name);
  void stopTimer(const std::string& name);
  double getTime(const std::string& name) const;
  void log(const std::string& name, double data);
private:
  std::map<std::string, Timer>      timers_;
  std::map<std::string, LogItem>    logs_;
  std::string                       trace_name_;        
  std::string                       trace_dir_;         
  std::ofstream                     ofs_;
  void trace();
  void traceHeader();
};
} 
