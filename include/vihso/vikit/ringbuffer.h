#pragma once
#include <vector>
#include <cassert>
#include <numeric>
namespace hso
{
template<typename T>
class RingBuffer
{
public:
  RingBuffer                 (int size);
  void
  push_back                  (const T & elem);
  bool
  empty                      () const;
  T
  get                        (int i);
  T
  getSum                     () const;
  T
  getMean                    () const;
  int size()
  {
    return num_elem_;
  }
private:
  std::vector<T> arr_;
  int begin_;
  int end_;
  int num_elem_;
  int arr_size_;
};
template <class T>
RingBuffer<T>
::RingBuffer(int size) :
    arr_(size),
    begin_(0),
    end_(-1),
    num_elem_(0),
    arr_size_(size)
{}
template <class T>
bool RingBuffer<T>
::empty() const
{
  return arr_.empty();
}
template <class T>
void RingBuffer<T>
::push_back(const T & elem)
{
  if (num_elem_<arr_size_)
  {
    end_++;
    arr_[end_] = elem;
    num_elem_++;
  }
  else{
    end_ = (end_+1)%arr_size_;
    begin_ = (begin_+1)%arr_size_;
    arr_[end_] = elem;
  }
}
template <class T>
T RingBuffer<T>
::get(int i)
{
  assert(i<num_elem_);
  return arr_[(begin_+i)%arr_size_];
}
template <class T>
T RingBuffer<T>
::getSum() const
{
  T sum=0;
  for(int i=0; i<num_elem_; ++i)
    sum+=arr_[i];
  return sum;
}
template <class T>
T RingBuffer<T>
::getMean() const
{
  if(num_elem_ == 0)
    return 0;
  return getSum()/num_elem_;
}
}
