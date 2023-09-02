#pragma once
#include <vihso/global.h>
#include <boost/thread.hpp>
#include <stdio.h>
#include <iostream>
#define MAPPING_THREADS 4
struct RunningStats
{
	int n_updates;			
	int n_failed_matches;	
	int n_out_views;
	int n_seeds;
	int n_fail_triangulation;	
	int n_fail_score;			
	int n_fail_alignment;		
	int n_fail_lsd;				
	inline RunningStats()
	{
		setZero();
	}
	inline void setZero()
	{
		memset(this,0,sizeof(RunningStats));
	}
	inline void add(RunningStats* r)
	{
		int* pt = (int*)this;
		int* pt_r = (int*)r;
		for(int i=0;i<static_cast<int>(sizeof(RunningStats)/sizeof(int));i++)
			pt[i] += pt_r[i];
	}
};
namespace lsd_slam
{
class IndexThreadReduce
{
public:
	inline IndexThreadReduce()
	{
		nextIndex = 0;
		maxIndex = 0;
		stepSize = 1;
		callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3);
		running = true;
		for(int i=0;i<MAPPING_THREADS;i++)
		{
			isDone[i] = false;
			workerThreads[i] = boost::thread(&IndexThreadReduce::workerLoop, this, i);
		}
	}
	inline ~IndexThreadReduce()
	{
		running = false;
		exMutex.lock();
		todo_signal.notify_all();
		exMutex.unlock();
		for(int i=0;i<MAPPING_THREADS;i++)
			workerThreads[i].join();
	}
	inline void reduce(boost::function<void(int,int,RunningStats*)> callPerIndex, int first, int end, RunningStats* stats, int stepSize = 0)
	{
		runningStats = stats;
		if(stepSize == 0)
			stepSize = ((end-first)+MAPPING_THREADS-1)/MAPPING_THREADS;
		boost::unique_lock<boost::mutex> lock(exMutex);
		this->callPerIndex = callPerIndex;
		nextIndex = first;
		maxIndex = end;
		this->stepSize = stepSize;
		for(int i=0;i<MAPPING_THREADS;i++)
			isDone[i] = false;
		todo_signal.notify_all();
		while(true)
		{
			done_signal.wait(lock);
			bool allDone = true;
			for(int i=0;i<MAPPING_THREADS;i++)
				allDone = allDone && isDone[i];
			if(allDone)
				break;
		}
		nextIndex = 0;
		maxIndex = 0;
		this->callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3);
	}
private:
	boost::thread workerThreads[MAPPING_THREADS];
	bool isDone[MAPPING_THREADS];
	boost::mutex exMutex;
	boost::condition_variable todo_signal;
	boost::condition_variable done_signal;
	int nextIndex;
	int maxIndex;
	int stepSize;
	RunningStats* runningStats;
	bool running;
	boost::function<void(int,int,RunningStats*)> callPerIndex;
	void callPerIndexDefault(int i, int j, RunningStats* k)
	{
		printf("ERROR: should never be called....\n");
	}
	void workerLoop(int idx)
	{
		boost::unique_lock<boost::mutex> lock(exMutex);
		while(running)
		{
			int todo = 0;
			bool gotSomething = false;
			if(nextIndex < maxIndex)
			{
				todo = nextIndex;
				nextIndex+=stepSize;
				gotSomething = true;
			}
			if(gotSomething)
			{
				lock.unlock();
				assert(callPerIndex != 0);
				RunningStats* s = new RunningStats();
				callPerIndex(todo, std::min(todo+stepSize, maxIndex), s);
				lock.lock();
				runningStats->add(s);
			}
			else
			{
				isDone[idx] = true;
				done_signal.notify_all();
				todo_signal.wait(lock);
			}
		}
	}
};
} 
