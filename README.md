# BTree implementation for Go

This package provides an in-memory B-Tree implementation for Go, useful as
an ordered, mutable data structure.

The API is based off of the wonderful
http://godoc.org/github.com/petar/GoLLRB/llrb, and is meant to allow btree to
act as a drop-in replacement for gollrb trees.

See http://godoc.org/github.com/google/btree for documentation.

## Overview ¶
Package btree implements in-memory B-Trees of arbitrary degree.
软件包 btree 实现了任意程度的内存 B 树。

btree implements an in-memory B-Tree for use as an ordered data structure. It is not meant for persistent storage solutions.
btree 实现了一个内存中的 B-Tree 有序数据结构。它不用于持久存储解决方案。

It has a flatter structure than an equivalent red-black or other binary tree, which in some cases yields better memory usage and/or performance. See some discussion on the matter here:
与等效的红黑树或其他二叉树相比，它的结构更扁平，在某些情况下可以获得更好的内存使用率和/或性能。有关这方面的讨论，请参见此处：

http://google-opensource.blogspot.com/2013/01/c-containers-that-save-memory-and-time.html
Note, though, that this project is in no way related to the C++ B-Tree implementation written about there.
但请注意，该项目与其中所写的 C++ B-Tree 实现完全无关。

Within this tree, each node contains a slice of items and a (possibly nil) slice of children. For basic numeric values or raw structs, this can cause efficiency differences when compared to equivalent C++ template code that stores values in arrays within the node:
在这棵树中，每个节点都包含一个项目片段和一个子项目片段（可能为零）。对于基本数值或原始结构体，这可能会导致与在节点内将数值存储在数组中的 C++ 模板代码相比效率上的差异：

Due to the overhead of storing values as interfaces (each value needs to be stored as the value itself, then 2 words for the interface pointing to that value and its type), resulting in higher memory use.
Since interfaces can point to values anywhere in memory, values are most likely not stored in contiguous blocks, resulting in a higher number of cache misses.
These issues don't tend to matter, though, when working with strings or other heap-allocated structures, since C++-equivalent structures also must store pointers and also distribute their values across the heap.
不过，在处理字符串或其他堆分配结构时，这些问题并不重要，因为 C++ 的等价结构也必须存储指针，并在堆上分配它们的值。

This implementation is designed to be a drop-in replacement to gollrb.LLRB trees, (http://github.com/petar/gollrb), an excellent and probably the most widely used ordered tree implementation in the Go ecosystem currently. Its functions, therefore, exactly mirror those of llrb.LLRB where possible. Unlike gollrb, though, we currently don't support storing multiple equivalent values.
本实现旨在直接替换 gollrb.LLRB 树 ( http://github.com/petar/gollrb)，后者是一个优秀的有序树实现，也可能是目前 Go 生态系统中使用最广泛的有序树实现。因此，它的功能尽可能完全照搬 llrb.LLRB 的功能。但与 gollrb 不同的是，我们目前不支持存储多个等价值。

There are two implementations; those suffixed with 'G' are generics, usable for any type, and require a passed-in "less" function to define their ordering. Those without this prefix are specific to the 'Item' interface, and use its 'Less' function for ordering.
有两种实现：后缀为 "G "的实现是泛型，可用于任何类型，并需要一个传入的 "Less "函数来定义其排序。而没有前缀的则是 "Item "接口专用的，并使用其 "Less "函数进行排序。