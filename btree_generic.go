// Copyright 2014-2022 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build go1.18
// +build go1.18

// 在Go 1.18及更高版本中，创建了一个 BTreeG 泛型，BTree 是针对 Item 接口的特定
// 实例化，具有向后兼容的 API。在go1.18之前，不支持泛型，
// BTree 仅是围绕 Item 接口的实现。

// Package btree btree 包实现了任意度的内存 B-Tree。
//
// btree 为有序数据结构实现了一个内存中的 B-Tree。
// 它并不适用于持久性存储解决方案。
//
// 与等效的红黑色或其他二叉树相比，它有一个更平坦的结构，
// 这在某些情况下可以产生更好的内存使用和/或性能。
// 有关此 matter的一些讨论在这里：
//
//	http://google-opensource.blogspot.com/2013/01/c-containers-that-save-memory-and-time.html
//
// 但是请注意，这个项目与那里写的 C++ B-Tree
// 实现没有任何关系。
//
// 在这棵树中，每个节点都包含一个条目的切片和一个（可能为空)
// 孩子的切片. 对于基本的数字值或原始结构，这可能在与等效的 C++
// 模板代码（在节点中存储值的数组）相比时导致效率差异:
//   - 由于存储值作为接口的开销（每个值都需要
//     以值本身存储，然后是 2 个字，用于指向该值及其类型的接口），导致
//     内存使用更高。
//   - 由于接口可以指向内存中的任何地方，值很可能不
//     存储在连续的块中，导致缓存未命中的数量增加。
//
// 然而，当处理字符串或其他堆分配的结构时，这些问题通常并不重要，
// 因为 C++ 等效结构也必须存储指针并在堆上分布它们的值。
//
// 该实现设计为 gollrb.LLRB 树的替代品（http://github.com/petar/gollrb），
// 这是一种出色的，可能是目前 Go 生态系统中最广泛使用的有序树实现。
// 因此，它的功能尽可能准确地反映了 llrb.LLRB。与 gollrb
// 不同的是，我们目前不支持存储多个等价值。
//
// 有两种实现；那些带有 'G' 后缀的是泛型，可用于任何类型，并且需要传入 "less" 函数以定义它们的排序。
// 没有此前缀的是针对 'Item' 接口的特定实现，并使用 'Less' 函数进行排序。
package btree

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
)

// Item 接口代表树中的单个对象。
type Item interface {
	// Less 方法用于测试当前项是否小于给定的参数。
	//
	// 这必须提供一个严格的弱序关系。
	// 如果 !a.Less(b) && !b.Less(a)，我们将此视为 a == b（即树中只能容纳 a 或 b 中的一个）。
	Less(than Item) bool
}

// 默认的 FreeList 大小
const (
	DefaultFreeListSize = 32
)

// FreeListG 表示 btree 节点的空闲列表。
// 默认情况下，每个 BTree 都有自己的 FreeList，
// 但多个 BTrees 可以共享同一个 FreeList，特别是当它们通过 Clone 创建时。
// 使用同一 freelist 的两个 Btrees 可以安全地进行并发写入访问。
type FreeListG[T any] struct {
	mu       sync.Mutex // 互斥锁，用于保证并发访问安全
	freelist []*node[T] // 存放空闲节点的切片
}

// NewFreeListG 创建一个新的空闲列表。
// size 是返回的空闲列表的最大大小。
func NewFreeListG[T any](size int) *FreeListG[T] {
	return &FreeListG[T]{freelist: make([]*node[T], 0, size)}
}

// FreeListG 中的 newNode 方法用于创建一个新的 node。
func (f *FreeListG[T]) newNode() (n *node[T]) {
	f.mu.Lock()                  // 锁定以保证线程安全
	index := len(f.freelist) - 1 // 获取 freelist 中最后一个元素的索引
	if index < 0 {
		f.mu.Unlock()       // 解锁
		return new(node[T]) // 如果 freelist 为空，创建一个新的 node 并返回
	}
	n = f.freelist[index]           // 从 freelist 获取一个现有的 node
	f.freelist[index] = nil         // 将获取的位置设为 nil
	f.freelist = f.freelist[:index] // 缩减 freelist
	f.mu.Unlock()                   // 解锁
	return
}

// freeNode 方法用于将不再使用的节点返回到 FreeListG。
func (f *FreeListG[T]) freeNode(n *node[T]) (out bool) {
	f.mu.Lock() // 锁定以保证线程安全
	if len(f.freelist) < cap(f.freelist) {
		f.freelist = append(f.freelist, n) // 将 node 添加到 freelist
		out = true                         // 设置返回值为 true 表示成功
	}
	f.mu.Unlock() // 解锁
	return
}

// ItemIteratorG 允许调用者在树的部分区域上按顺序迭代。
// 当此函数返回 false 时，迭代将停止，并且关联的 Ascend* 函数将立即返回。
type ItemIteratorG[T any] func(item T) bool

// Ordered 表示一组类型，对于这些类型 '<' 操作符有效。
// 它包含了基本的整数、无符号整数、浮点数和字符串类型。
type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64 | ~string
}

// Less[T] 返回一个默认的 LessFunc，该函数对支持 '<' 操作符的类型使用 '<' 进行比较。
func Less[T Ordered]() LessFunc[T] {
	return func(a, b T) bool { return a < b } // 使用 '<' 操作符比较 a 和 b
}

// NewOrderedG 为有序类型创建一个新的 B-树。
func NewOrderedG[T Ordered](degree int) *BTreeG[T] {
	return NewG[T](degree, Less[T]()) // 使用默认的 LessFunc 创建 B-树
}

// NewG 创建一个具有给定度数的新 B-树。
//
// 例如，NewG(2) 将创建一个 2-3-4 树（每个节点包含 1-3 个项和 2-4 个子节点）。
//
// 传入的 LessFunc 决定了类型 T 的对象是如何排序的。
func NewG[T any](degree int, less LessFunc[T]) *BTreeG[T] {
	// 使用默认的 FreeList 大小创建 B-树
	return NewWithFreeListG(degree, less, NewFreeListG[T](DefaultFreeListSize))
}

// NewWithFreeListG 创建一个使用给定节点空闲列表的新 B-树。
func NewWithFreeListG[T any](degree int, less LessFunc[T], f *FreeListG[T]) *BTreeG[T] {
	if degree <= 1 {
		panic("bad degree") // 如果度数小于等于 1，则抛出异常
	}
	// 创建并返回一个 BTreeG 实例
	return &BTreeG[T]{
		degree: degree,                                          // 设置度数
		cow:    &copyOnWriteContext[T]{freelist: f, less: less}, // 设置写时复制上下文
	}
}

// items 类型用于存储节点中的元素。
type items[T any] []T

// insertAt 方法在给定索引处插入一个值，将所有后续值向后推移。
func (s *items[T]) insertAt(index int, item T) {
	var zero T
	*s = append(*s, zero) // 在末尾添加一个零值以扩展切片
	if index < len(*s) {
		copy((*s)[index+1:], (*s)[index:]) // 将从 index 开始的元素向后移动一个位置
	}
	(*s)[index] = item // 在指定索引处插入新元素
}

// removeAt 方法移除给定索引处的值，并将所有后续值向前拉。
func (s *items[T]) removeAt(index int) T {
	item := (*s)[index]                // 保存被移除的元素
	copy((*s)[index:], (*s)[index+1:]) // 将从 index+1 开始的元素向前移动一个位置
	var zero T
	(*s)[len(*s)-1] = zero // 清除最后一个元素
	*s = (*s)[:len(*s)-1]  // 缩减切片大小
	return item            // 返回被移除的元素
}

// pop 方法移除并返回列表中的最后一个元素。
func (s *items[T]) pop() (out T) {
	index := len(*s) - 1 // 获取最后一个元素的索引
	out = (*s)[index]    // 保存最后一个元素
	var zero T
	(*s)[index] = zero // 清除最后一个元素
	*s = (*s)[:index]  // 缩减切片大小
	return             // 返回被移除的元素
}

// truncate 方法将此实例在 index 处截断，使其仅包含前 index 个项。
// index 必须小于或等于长度。
func (s *items[T]) truncate(index int) {
	var toClear items[T]
	// 将 s 切片分割为两部分，一部分是截断后的 s，另一部分是要清除的部分
	*s, toClear = (*s)[:index], (*s)[index:]
	var zero T
	// 清除 toClear 切片中的所有元素
	for i := 0; i < len(toClear); i++ {
		toClear[i] = zero
	}
}

// find 方法返回给定项应该插入到这个列表中的索引。
// 如果列表中已经存在该项，则 'found' 为 true。
func (s items[T]) find(item T, less func(T, T) bool) (index int, found bool) {
	i := sort.Search(len(s), func(i int) bool {
		return less(item, s[i])
	})
	// 检查找到的位置的前一个位置是否是要找的项
	if i > 0 && !less(s[i-1], item) {
		return i - 1, true // 找到项
	}
	return i, false // 没有找到项
}

// node 是树中的内部节点。
//
// 它必须始终维持以下不变量：
//   - 如果 len(children) == 0，则 len(items) 没有限制
//   - 如果 len(children) != 0，则 len(children) == len(items) + 1
type node[T any] struct {
	items    items[T]               // 存储在节点中的元素
	children items[*node[T]]        // 子节点的切片
	cow      *copyOnWriteContext[T] // 指向写时复制上下文的指针
}

// mutableFor 方法确保节点是可变的。
// 如果节点已经属于给定的写时复制上下文，则直接返回该节点；
// 否则，创建一个新节点并复制数据。
func (n *node[T]) mutableFor(cow *copyOnWriteContext[T]) *node[T] {
	// 如果当前节点已经属于给定的写时复制上下文，直接返回该节点
	if n.cow == cow {
		return n
	}
	// 否则，创建一个新的节点
	out := cow.newNode()
	// 根据需要的容量初始化或扩展 out.items
	if cap(out.items) >= len(n.items) {
		out.items = out.items[:len(n.items)]
	} else {
		out.items = make(items[T], len(n.items), cap(n.items))
	}
	// 复制 items 到新节点
	copy(out.items, n.items)
	// 复制 children
	// 根据需要的容量初始化或扩展 out.children
	if cap(out.children) >= len(n.children) {
		out.children = out.children[:len(n.children)]
	} else {
		out.children = make(items[*node[T]], len(n.children), cap(n.children))
	}
	// 复制 children 到新节点
	copy(out.children, n.children)
	return out
}

// mutableChild 方法确保给定索引处的子节点是可变的，并返回该节点。
func (n *node[T]) mutableChild(i int) *node[T] {
	c := n.children[i].mutableFor(n.cow) // 确保子节点是可变的
	n.children[i] = c                    // 更新子节点
	return c                             // 返回更新后的子节点
}

// split 方法在给定索引处拆分节点。当前节点将缩小，此函数返回该索引处的项
// 以及一个新节点，包含该索引之后的所有项/子节点。
func (n *node[T]) split(i int) (T, *node[T]) {
	item := n.items[i]                                // 获取拆分点的项
	next := n.cow.newNode()                           // 创建一个新节点
	next.items = append(next.items, n.items[i+1:]...) // 将拆分点之后的项添加到新节点
	n.items.truncate(i)                               // 截断当前节点在拆分点的项

	// 如果有子节点，处理子节点
	if len(n.children) > 0 {
		next.children = append(next.children, n.children[i+1:]...) // 将拆分点之后的子节点添加到新节点
		n.children.truncate(i + 1)                                 // 截断当前节点在拆分点的子节点
	}
	return item, next // 返回拆分项和新节点
}

// maybeSplitChild 方法检查是否应该拆分一个子节点，如果是，则拆分它。
// 返回是否发生了拆分。
func (n *node[T]) maybeSplitChild(i, maxItems int) bool {
	// 如果子节点的项数小于最大项数，不进行拆分
	if len(n.children[i].items) < maxItems {
		return false
	}
	first := n.mutableChild(i)                // 获取可变的子节点
	item, second := first.split(maxItems / 2) // 拆分子节点
	n.items.insertAt(i, item)                 // 在当前节点插入拆分项
	n.children.insertAt(i+1, second)          // 在当前节点插入新生成的子节点
	return true                               // 返回 true 表示发生了拆分
}

// insert 方法将一个项插入以这个节点为根的子树中，确保子树中的任何节点都不超过 maxItems 个项。
// 如果找到相同的项被插入或替换，它将被返回。
func (n *node[T]) insert(item T, maxItems int) (_ T, _ bool) {
	i, found := n.items.find(item, n.cow.less) // 在 items 中查找 item，返回索引和是否找到
	if found {
		out := n.items[i] // 如果找到，保存旧值
		n.items[i] = item // 替换为新项
		return out, true  // 返回旧值和 true
	}
	if len(n.children) == 0 {
		n.items.insertAt(i, item) // 如果没有子节点，直接插入项
		return
	}
	if n.maybeSplitChild(i, maxItems) { // 如果需要，拆分子节点
		inTree := n.items[i] // 拆分后的项
		switch {
		case n.cow.less(item, inTree):
			// 无变化，我们要的是第一个拆分节点
		case n.cow.less(inTree, item):
			i++ // 我们要的是第二个拆分节点
		default:
			out := n.items[i] // 保存旧值
			n.items[i] = item // 替换为新项
			return out, true  // 返回旧值和 true
		}
	}
	// 递归插入到子节点
	return n.mutableChild(i).insert(item, maxItems)
}

// get 方法在以该节点为根的子树中查找给定的键，并返回它。
func (n *node[T]) get(key T) (_ T, _ bool) {
	i, found := n.items.find(key, n.cow.less) // 在 items 中查找 key，返回索引和是否找到
	if found {
		return n.items[i], true // 如果找到，返回找到的元素和 true
	} else if len(n.children) > 0 {
		return n.children[i].get(key) // 如果没有找到且有子节点，递归搜索子节点
	}
	return // 如果没有找到，返回零值和 false
}

// min 函数返回子树中的第一个元素。
func min[T any](n *node[T]) (_ T, found bool) {
	if n == nil {
		return // 如果节点为 nil，返回零值和 false
	}
	for len(n.children) > 0 {
		n = n.children[0] // 沿最左侧路径向下遍历
	}
	if len(n.items) == 0 {
		return // 如果没有元素，返回零值和 false
	}
	return n.items[0], true // 返回最左侧元素和 true
}

// max 函数返回子树中的最后一个元素。
func max[T any](n *node[T]) (_ T, found bool) {
	if n == nil {
		return // 如果节点为 nil，返回零值和 false
	}
	for len(n.children) > 0 {
		n = n.children[len(n.children)-1] // 沿最右侧路径向下遍历
	}
	if len(n.items) == 0 {
		return // 如果没有元素，返回零值和 false
	}
	return n.items[len(n.items)-1], true // 返回最右侧元素和 true
}

// toRemove 类型详细说明了在 node.remove 调用中要移除的项目。
type toRemove int

const (
	removeItem toRemove = iota // 移除给定的项
	removeMin                  // 移除子树中最小的项
	removeMax                  // 移除子树中最大的项
)

// remove 方法从以这个节点为根的子树中移除一个项。
func (n *node[T]) remove(item T, minItems int, typ toRemove) (_ T, _ bool) {
	var i int
	var found bool
	switch typ {
	case removeMax:
		if len(n.children) == 0 {
			// 如果没有子节点，直接移除并返回最后一个项
			return n.items.pop(), true
		}
		i = len(n.items) // 设定索引为最后一个项的位置
	case removeMin:
		if len(n.children) == 0 {
			// 如果没有子节点，直接移除并返回第一个项
			return n.items.removeAt(0), true
		}
		i = 0 // 设定索引为第一个项的位置
	case removeItem:
		i, found = n.items.find(item, n.cow.less) // 在 items 中查找 item
		if len(n.children) == 0 {
			// 如果没有子节点且找到了项，直接移除并返回该项
			if found {
				return n.items.removeAt(i), true
			}
			return // 如果没有找到，返回零值和 false
		}
	default:
		panic("invalid type") // 如果 typ 不合法，抛出异常
	}
	// 如果有子节点
	if len(n.children[i].items) <= minItems {
		// 如果子节点项数不足，调用 growChildAndRemove 方法
		return n.growChildAndRemove(i, item, minItems, typ)
	}
	child := n.mutableChild(i)
	// 此时子节点有足够的项进行操作
	if found {
		// 如果在当前节点找到了项
		out := n.items[i]
		// 特殊情况下，使用 removeMax 来获取前驱并替换当前项
		var zero T
		n.items[i], _ = child.remove(zero, minItems, removeMax)
		return out, true
	}
	// 递归调用 remove 方法
	// 此时我们知道项不在当前节点且子节点足够大以移除项
	return child.remove(item, minItems, typ)
}

// growChildAndRemove 方法确保子节点 'i' 足够大以便在保持最小项数的同时能够从中移除一个项，
// 然后调用 remove 来实际移除它。
//
// 大多数文档表示我们需要处理两组特殊情况：
//  1. 项在这个节点中
//  2. 项在子节点中
//
// 在这两种情况下，我们需要处理以下两个子情况：
//
//	A) 节点有足够的值可以剥离一个
//	B) 节点没有足够的值
//
// 对于后者，我们需要检查：
//
//	a) 左侧兄弟节点有多余的节点可用
//	b) 右侧兄弟节点有多余的节点可用
//	c) 我们必须合并
//
// 为了简化我们的代码，我们将情况 #1 和 #2 以相同的方式处理：
// 如果一个节点没有足够的项，我们确保它有（使用 a、b、c）。
// 然后我们重新进行我们的 remove 调用，第二次（无论我们是否在情况 1 或 2 中），
// 我们将拥有足够的项，并可以保证我们触及情况 A。
func (n *node[T]) growChildAndRemove(i int, item T, minItems int, typ toRemove) (T, bool) {
	if i > 0 && len(n.children[i-1].items) > minItems {
		// 从左侧子节点“窃取”一个项
		child := n.mutableChild(i)
		stealFrom := n.mutableChild(i - 1)
		stolenItem := stealFrom.items.pop()   // 弹出左侧子节点的最后一个项
		child.items.insertAt(0, n.items[i-1]) // 将当前节点的项插入到右侧子节点的起始位置
		n.items[i-1] = stolenItem             // 将被窃取的项放入当前节点

		// 如果有子节点，也进行窃取
		if len(stealFrom.children) > 0 {
			child.children.insertAt(0, stealFrom.children.pop())
		}
	} else if i < len(n.items) && len(n.children[i+1].items) > minItems {
		// 从右侧子节点“窃取”一个项
		child := n.mutableChild(i)
		stealFrom := n.mutableChild(i + 1)
		stolenItem := stealFrom.items.removeAt(0) // 移除右侧子节点的第一个项
		child.items = append(child.items, n.items[i])
		n.items[i] = stolenItem // 将被窃取的项放入当前节点

		// 如果有子节点，也进行窃取
		if len(stealFrom.children) > 0 {
			child.children = append(child.children, stealFrom.children.removeAt(0))
		}
	} else {
		// 如果无法窃取，进行合并
		if i >= len(n.items) {
			i--
		}
		child := n.mutableChild(i)
		mergeItem := n.items.removeAt(i) // 移除当前节点的项，准备合并
		mergeChild := n.children.removeAt(i + 1)
		// 将项和子节点合并到左侧子节点
		child.items = append(child.items, mergeItem)
		child.items = append(child.items, mergeChild.items...)
		child.children = append(child.children, mergeChild.children...)
		n.cow.freeNode(mergeChild) // 释放被合并的子节点
	}
	// 调用 remove 移除项
	return n.remove(item, minItems, typ)
}

// direction 类型表示遍历的方向。
type direction int

const (
	descend = direction(-1) // 降序
	ascend  = direction(+1) // 升序
)

// optionalItem 类型封装了一个可能不存在的项。
type optionalItem[T any] struct {
	item  T    // 封装的项
	valid bool // 项是否有效（是否存在）
}

// optional 函数创建一个有效的 optionalItem。
func optional[T any](item T) optionalItem[T] {
	return optionalItem[T]{item: item, valid: true}
}

// empty 函数创建一个无效的 optionalItem。
func empty[T any]() optionalItem[T] {
	return optionalItem[T]{}
}

// iterate 提供了一种简单的方法来遍历树中的元素。
//
// 当升序时，'start' 应小于 'stop'；当降序时，'start' 应大于 'stop'。
// 将 'includeStart' 设置为 true 将迫使迭代器在第一个项等于 'start' 时包括它，
// 从而创建一个 "greaterOrEqual" 或 "lessThanEqual" 查询，而不仅仅是
// "greaterThan" 或 "lessThan" 查询。
func (n *node[T]) iterate(dir direction, start, stop optionalItem[T], includeStart bool, hit bool, iter ItemIteratorG[T]) (bool, bool) {
	var ok, found bool
	var index int
	switch dir {
	case ascend: // 升序
		// 如果提供了起始点并且它是有效的，则找到起始点的索引
		if start.valid {
			index, _ = n.items.find(start.item, n.cow.less)
		}
		// 从起始点开始遍历节点的每个项
		for i := index; i < len(n.items); i++ {
			// 如果存在子节点，则对子节点进行递归迭代
			if len(n.children) > 0 {
				if hit, ok = n.children[i].iterate(dir, start, stop, includeStart, hit, iter); !ok {
					return hit, false
				}
			}
			// 如果不包括起始点，并且未命中起始点，并且起始点有效，检查当前项是否为起始点
			if !includeStart && !hit && start.valid && !n.cow.less(start.item, n.items[i]) {
				hit = true // 标记已命中起始点
				continue   // 跳过当前项，继续下一个项
			}
			hit = true // 标记已命中起始点
			// 如果提供了结束点并且当前项不小于结束点，结束迭代
			if stop.valid && !n.cow.less(n.items[i], stop.item) {
				return hit, false
			}
			// 调用迭代器函数，如果返回 false，则结束迭代
			if !iter(n.items[i]) {
				return hit, false
			}
		}
		// 如果存在子节点，对最后一个子节点进行递归迭代
		if len(n.children) > 0 {
			if hit, ok = n.children[len(n.children)-1].iterate(dir, start, stop, includeStart, hit, iter); !ok {
				return hit, false
			}
		}
	case descend: // 降序
		// 如果提供了起始点并且它是有效的，则找到起始点的索引
		if start.valid {
			index, found = n.items.find(start.item, n.cow.less) // 找到起始点
			if !found {
				index = index - 1 // 如果没有找到精确匹配的项，则移动到前一个项
			}
		} else {
			index = len(n.items) - 1 // 如果没有提供起始点，则从最后一个项开始
		}
		// 从找到的起始点开始，向前遍历节点的每个项
		for i := index; i >= 0; i-- {
			// 检查是否满足起始条件
			if start.valid && !n.cow.less(n.items[i], start.item) {
				if !includeStart || hit || n.cow.less(start.item, n.items[i]) {
					continue // 如果不包括起始点或已经处理过起始点，则跳过
				}
			}
			// 遍历子节点
			if len(n.children) > 0 {
				if hit, ok = n.children[i+1].iterate(dir, start, stop, includeStart, hit, iter); !ok {
					return hit, false
				}
			}
			// 检查是否到达结束点
			if stop.valid && !n.cow.less(stop.item, n.items[i]) {
				return hit, false // 如果当前项大于或等于结束点，则停止迭代
			}
			hit = true
			// 执行迭代器函数
			if !iter(n.items[i]) {
				return hit, false // 如果迭代器函数返回 false，则停止迭代
			}
		}
		// 如果存在子节点，对第一个子节点进行递归迭代
		if len(n.children) > 0 {
			if hit, ok = n.children[0].iterate(dir, start, stop, includeStart, hit, iter); !ok {
				return hit, false
			}
		}
	}
	return hit, true
}

// print is used for testing/debugging purposes.
func (n *node[T]) print(w io.Writer, level int) {
	fmt.Fprintf(w, "%sNODE:%v\n", strings.Repeat("  ", level), n.items)
	for _, c := range n.children {
		c.print(w, level+1)
	}
}

// BTreeG 是 B-树的泛型实现。
//
// BTreeG 存储类型为 T 的项，提供有序结构，便于插入、删除和迭代。
//
// 写操作对于多个 goroutines 的并发修改是不安全的，但读操作是安全的。
type BTreeG[T any] struct {
	degree int                    // 树的度数，决定了节点的子节点数量
	length int                    // 树中元素的总数
	root   *node[T]               // 树的根节点
	cow    *copyOnWriteContext[T] // 写时复制上下文
}

// LessFunc LessFunc[T] 用于确定如何对类型 'T' 进行排序。
// 它应该实现一个严格的排序规则，如果在该排序中 'a' < 'b'，则应返回 true。
type LessFunc[T any] func(a, b T) bool

// copyOnWriteContext 指针用于确定节点的所有权...
// 如果一个树的写入上下文与节点的写入上下文相同，那么它被允许修改该节点。
// 如果一个树的写入上下文与节点的不匹配，它不被允许修改该节点，
// 并且必须创建一个新的、可写的副本（即：它是一个克隆）。
//
// 在执行任何写操作时，我们维持当前节点的上下文与请求写操作的树的上下文相等的不变量。
// 我们通过在下降到任何节点之前，如果上下文不匹配，创建一个具有正确上下文的副本来做到这一点。
//
// 由于我们在任何写操作中当前访问的节点具有请求树的上下文，那么该节点可以就地修改。
// 该节点的子节点可能不共享上下文，但在我们下降到它们之前，我们将制作一个可变副本。
type copyOnWriteContext[T any] struct {
	freelist *FreeListG[T] // 指向 FreeListG 的指针，用于管理空闲的节点
	less     LessFunc[T]   // LessFunc 类型的函数，用于确定元素的排序
}

// Clone 方法懒惰地克隆 B-树。
// Clone 不应该并发调用，但一旦 Clone 调用完成，原始树（t）和新树（t2）可以并发使用。
//
// b 的内部树结构被标记为只读并在 t 和 t2 之间共享。
// 对 t 和 t2 的写操作使用写时复制逻辑，在 b 的原始节点将被修改时创建新节点。
// 读操作性能不会下降。t 和 t2 的写操作最初会因额外的分配和复制而稍微变慢，
// 但应该逐渐恢复到原始树的性能特征。
func (t *BTreeG[T]) Clone() (t2 *BTreeG[T]) {
	// 创建两个全新的写时复制上下文。
	// 这个操作实际上创建了三棵树：
	//   原始的，共享节点（旧的 b.cow）
	//   新的 b.cow 节点
	//   新的 out.cow 节点
	cow1, cow2 := *t.cow, *t.cow
	out := *t
	t.cow = &cow1
	out.cow = &cow2
	return &out
}

// maxItems 方法返回每个节点允许的最大项数。
func (t *BTreeG[T]) maxItems() int {
	return t.degree*2 - 1
}

// minItems 方法返回每个节点允许的最小项数（对根节点忽略）。
func (t *BTreeG[T]) minItems() int {
	return t.degree - 1
}

// newNode 方法创建一个新的节点。
func (c *copyOnWriteContext[T]) newNode() (n *node[T]) {
	n = c.freelist.newNode() // 从 freelist 中获取一个新节点
	n.cow = c                // 设置新节点的写时复制上下文
	return
}

type freeType int

// 定义 freeType 类型和相关的常量，用于表示释放节点的不同情况。
const (
	ftFreelistFull freeType = iota // 节点已被释放（可被垃圾回收，不存储在 freelist 中）
	ftStored                       // 节点被存储在 freelist 中，以供后续使用
	ftNotOwned                     // 节点被写时复制（COW）忽略，因为它由另一个上下文拥有
)

// freeNode 在给定的写时复制上下文中释放一个节点，如果它由该上下文拥有。
// 它返回节点发生的情况（参见 freeType 常量文档）。
func (c *copyOnWriteContext[T]) freeNode(n *node[T]) freeType {
	if n.cow == c {
		// 清除以允许垃圾回收
		n.items.truncate(0)
		n.children.truncate(0)
		n.cow = nil
		// 尝试将节点存储到 freelist
		if c.freelist.freeNode(n) {
			return ftStored
		} else {
			return ftFreelistFull
		}
	} else {
		// 如果节点不由当前上下文拥有，返回 ftNotOwned
		return ftNotOwned
	}
}

// ReplaceOrInsert 方法向树中添加给定的项。
// 如果树中已经有一个等于给定项的元素，它将从树中移除并返回，
// 第二个返回值为 true。否则，返回 (zeroValue, false)。
//
// 不能向树中添加 nil（会导致 panic）。
func (t *BTreeG[T]) ReplaceOrInsert(item T) (_ T, _ bool) {
	if t.root == nil {
		// 如果树是空的，创建新的根节点并添加项
		t.root = t.cow.newNode()
		t.root.items = append(t.root.items, item)
		t.length++ // 增加树的长度
		return
	} else {
		// 确保根节点是可变的
		t.root = t.root.mutableFor(t.cow)
		// 如果根节点的项数达到最大限制
		if len(t.root.items) >= t.maxItems() {
			// 拆分根节点
			item2, second := t.root.split(t.maxItems() / 2)
			oldroot := t.root
			// 创建新的根节点
			t.root = t.cow.newNode()
			t.root.items = append(t.root.items, item2)
			t.root.children = append(t.root.children, oldroot, second)
		}
	}
	// 插入项
	out, outb := t.root.insert(item, t.maxItems())
	if !outb {
		t.length++ // 如果插入新项，增加树的长度
	}
	return out, outb // 返回结果
}

// Delete removes an item equal to the passed in item from the tree, returning
// it.  If no such item exists, returns (zeroValue, false).
func (t *BTreeG[T]) Delete(item T) (T, bool) {
	return t.deleteItem(item, removeItem)
}

// DeleteMin 移除并返回树中最小的项。
// 如果树为空，返回 (zeroValue, false)。
func (t *BTreeG[T]) DeleteMin() (T, bool) {
	var zero T
	// 调用 deleteItem 函数删除最小项
	return t.deleteItem(zero, removeMin)
}

// DeleteMax 移除并返回树中最大的项。
// 如果树为空，返回 (zeroValue, false)。
func (t *BTreeG[T]) DeleteMax() (T, bool) {
	var zero T
	// 调用 deleteItem 函数删除最大项
	return t.deleteItem(zero, removeMax)
}

// deleteItem 辅助函数，用于删除指定类型的项（最小或最大）。
func (t *BTreeG[T]) deleteItem(item T, typ toRemove) (_ T, _ bool) {
	if t.root == nil || len(t.root.items) == 0 {
		// 如果树为空，返回零值和 false
		return
	}
	// 确保根节点是可变的
	t.root = t.root.mutableFor(t.cow)
	// 调用 root 的 remove 方法删除项
	out, outb := t.root.remove(item, t.minItems(), typ)
	if len(t.root.items) == 0 && len(t.root.children) > 0 {
		// 如果根节点没有项但有子节点，将第一个子节点提升为新的根节点
		oldroot := t.root
		t.root = t.root.children[0]
		t.cow.freeNode(oldroot) // 释放旧的根节点
	}
	if outb {
		t.length-- // 如果成功删除了项，减少树的长度
	}
	return out, outb // 返回结果
}

// AscendRange 为树中范围 [greaterOrEqual, lessThan) 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTreeG[T]) AscendRange(greaterOrEqual, lessThan T, iterator ItemIteratorG[T]) {
	if t.root == nil {
		return // 如果根节点为空，直接返回
	}
	// 以升序方式迭代，包括起始点
	t.root.iterate(ascend, optional[T](greaterOrEqual), optional[T](lessThan), true, false, iterator)
}

// AscendLessThan 为树中范围 [first, pivot) 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTreeG[T]) AscendLessThan(pivot T, iterator ItemIteratorG[T]) {
	if t.root == nil {
		return // 如果根节点为空，直接返回
	}
	// 以升序方式迭代，不包括起始点
	t.root.iterate(ascend, empty[T](), optional(pivot), false, false, iterator)
}

// AscendGreaterOrEqual 为树中范围 [pivot, last] 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTreeG[T]) AscendGreaterOrEqual(pivot T, iterator ItemIteratorG[T]) {
	if t.root == nil {
		return // 如果根节点为空，直接返回
	}
	// 以升序方式迭代，包括起始点
	t.root.iterate(ascend, optional[T](pivot), empty[T](), true, false, iterator)
}

// Ascend 调用迭代器为树中范围 [first, last] 内的每个值进行遍历，直到迭代器返回 false。
func (t *BTreeG[T]) Ascend(iterator ItemIteratorG[T]) {
	if t.root == nil {
		return // 如果根节点为空，直接返回
	}
	// 以升序方式迭代整棵树
	t.root.iterate(ascend, empty[T](), empty[T](), false, false, iterator)
}

// DescendRange 为树中范围 [lessOrEqual, greaterThan) 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTreeG[T]) DescendRange(lessOrEqual, greaterThan T, iterator ItemIteratorG[T]) {
	if t.root == nil {
		return // 如果根节点为空，直接返回
	}
	// 以降序方式迭代，包括起始点
	t.root.iterate(descend, optional[T](lessOrEqual), optional[T](greaterThan), true, false, iterator)
}

// DescendLessOrEqual 为树中范围 [pivot, first] 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTreeG[T]) DescendLessOrEqual(pivot T, iterator ItemIteratorG[T]) {
	if t.root == nil {
		return // 如果根节点为空，直接返回
	}
	// 以降序方式迭代，包括起始点
	t.root.iterate(descend, optional[T](pivot), empty[T](), true, false, iterator)
}

// DescendGreaterThan 为树中范围 [last, pivot) 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTreeG[T]) DescendGreaterThan(pivot T, iterator ItemIteratorG[T]) {
	if t.root == nil {
		return // 如果根节点为空，直接返回
	}
	// 以降序方式迭代，不包括起始点
	t.root.iterate(descend, empty[T](), optional[T](pivot), false, false, iterator)
}

// Descend 调用迭代器为树中范围 [last, first] 内的每个值进行遍历，直到迭代器返回 false。
func (t *BTreeG[T]) Descend(iterator ItemIteratorG[T]) {
	if t.root == nil {
		return // 如果根节点为空，直接返回
	}
	// 以降序方式迭代整棵树
	t.root.iterate(descend, empty[T](), empty[T](), false, false, iterator)
}

// Get looks for the key item in the tree, returning it.  It returns
// (zeroValue, false) if unable to find that item.
func (t *BTreeG[T]) Get(key T) (_ T, _ bool) {
	if t.root == nil {
		return
	}
	return t.root.get(key)
}

// Min returns the smallest item in the tree, or (zeroValue, false) if the tree is empty.
func (t *BTreeG[T]) Min() (_ T, _ bool) {
	return min(t.root)
}

// Max returns the largest item in the tree, or (zeroValue, false) if the tree is empty.
func (t *BTreeG[T]) Max() (_ T, _ bool) {
	return max(t.root)
}

// Has returns true if the given key is in the tree.
func (t *BTreeG[T]) Has(key T) bool {
	_, ok := t.Get(key)
	return ok
}

// Len returns the number of items currently in the tree.
func (t *BTreeG[T]) Len() int {
	return t.length
}

// Clear 方法移除 btree 中的所有项。
// 如果 addNodesToFreelist 为 true，那么 t 的节点将作为这个调用的一部分被添加到它的 freelist 中，
// 直到 freelist 被填满。否则，简单地取消引用根节点，并将子树留给 Go 的常规垃圾回收过程。
//
// 这种方式比对所有元素调用 Delete 快得多，因为 Delete 需要在树中找到/移除
// 每个元素，并相应地更新树。它也比创建一个新树来替换旧树稍微快一些，
// 因为旧树的节点被回收到 freelist 中以供新树使用，而不是丢给垃圾回收器。
//
// 这个调用的时间复杂度：
//
//	O(1): 当 addNodesToFreelist 为 false 时，这是一个单一操作。
//	O(1): 当 freelist 已经满了，它会立即退出。
//	O(freelist size): 当 freelist 为空且所有节点都由这棵树拥有时，
//	   节点会被添加到 freelist 中，直到它被填满。
//	O(tree size): 当所有节点都由另一棵树拥有时，会迭代所有节点以寻找可添加到 freelist 的节点，
//	   由于所有权问题，没有节点会被添加。
func (t *BTreeG[T]) Clear(addNodesToFreelist bool) {
	if t.root != nil && addNodesToFreelist {
		t.root.reset(t.cow) // 重置根节点及其所有子节点，将它们添加到 freelist
	}
	t.root, t.length = nil, 0 // 清空树的根节点和长度
}

// reset 方法将子树返回到 freelist。
// 如果 freelist 已满，它会立即中断，因为迭代的唯一好处就是填满 freelist。
// 如果父节点的 reset 调用应该继续，则返回 true。
func (n *node[T]) reset(c *copyOnWriteContext[T]) bool {
	for _, child := range n.children {
		// 对所有子节点递归调用 reset
		if !child.reset(c) {
			return false
		}
	}
	// 尝试将当前节点添加到 freelist，如果 freelist 已满，返回 false
	return c.freeNode(n) != ftFreelistFull
}

// Int implements the Item interface for integers.
type Int int

// Less returns true if int(a) < int(b).
func (a Int) Less(b Item) bool {
	return a < b.(Int)
}

// BTree 是 B-树的实现。
//
// BTree 在有序结构中存储 Item 实例，便于插入、删除和迭代。
//
// 写操作在多个 goroutines 的并发修改下是不安全的，但读操作是安全的。
type BTree BTreeG[Item]

// itemLess 是一个 LessFunc，用于比较两个 Item 实例。
var itemLess LessFunc[Item] = func(a, b Item) bool {
	return a.Less(b)
}

// New 创建一个具有给定度数的新 B-树。
//
// 例如，New(2) 将创建一个 2-3-4 树（每个节点包含 1-3 个项和 2-4 个子节点）。
func New(degree int) *BTree {
	// 使用 Item 类型和 itemLess 比较函数创建新的泛型 B-树
	return (*BTree)(NewG[Item](degree, itemLess))
}

// FreeList 表示 btree 节点的空闲列表。
// 默认情况下，每个 BTree 都有自己的 FreeList，但多个 BTrees 可以共享同一个 FreeList。
// 使用同一个 freelist 的两个 Btrees 可以安全地进行并发写入访问。
type FreeList FreeListG[Item]

// NewFreeList 创建一个新的空闲列表。
// size 是返回的空闲列表的最大大小。
func NewFreeList(size int) *FreeList {
	// 使用 Item 类型创建一个新的泛型空闲列表
	return (*FreeList)(NewFreeListG[Item](size))
}

// NewWithFreeList 创建一个使用给定节点空闲列表的新 B-树。
func NewWithFreeList(degree int, f *FreeList) *BTree {
	// 使用给定的 FreeList 创建一个新的 B-树
	return (*BTree)(NewWithFreeListG[Item](degree, itemLess, (*FreeListG[Item])(f)))
}

// ItemIterator 允许调用 Ascend* 函数的调用者按顺序在树的部分区域上迭代。
// 当这个函数返回 false 时，迭代将停止，相关的 Ascend* 函数将立即返回。
type ItemIterator ItemIteratorG[Item]

// Clone 懒惰地克隆 btree。Clone 不应并发调用，
// 但一旦 Clone 调用完成，原始树（t）和新树（t2）可以并发使用。
//
// b 的内部树结构被标记为只读并在 t 和 t2 之间共享。
// 对 t 和 t2 的写操作使用写时复制逻辑，当 b 的原始节点将被修改时创建新节点。
// 读操作性能不会降低。t 和 t2 的写操作最初会因额外的分配和复制而稍微变慢，
// 但应该逐渐恢复到原始树的性能特征。
func (t *BTree) Clone() (t2 *BTree) {
	// 调用泛型 B-树的 Clone 方法并将结果转换为 BTree 类型
	return (*BTree)((*BTreeG[Item])(t).Clone())
}

// Delete removes an item equal to the passed in item from the tree, returning
// it.  If no such item exists, returns nil.
func (t *BTree) Delete(item Item) Item {
	i, _ := (*BTreeG[Item])(t).Delete(item)
	return i
}

// DeleteMax removes the largest item in the tree and returns it.
// If no such item exists, returns nil.
func (t *BTree) DeleteMax() Item {
	i, _ := (*BTreeG[Item])(t).DeleteMax()
	return i
}

// DeleteMin removes the smallest item in the tree and returns it.
// If no such item exists, returns nil.
func (t *BTree) DeleteMin() Item {
	i, _ := (*BTreeG[Item])(t).DeleteMin()
	return i
}

// Get looks for the key item in the tree, returning it.  It returns nil if
// unable to find that item.
func (t *BTree) Get(key Item) Item {
	i, _ := (*BTreeG[Item])(t).Get(key)
	return i
}

// Max returns the largest item in the tree, or nil if the tree is empty.
func (t *BTree) Max() Item {
	i, _ := (*BTreeG[Item])(t).Max()
	return i
}

// Min returns the smallest item in the tree, or nil if the tree is empty.
func (t *BTree) Min() Item {
	i, _ := (*BTreeG[Item])(t).Min()
	return i
}

// Has returns true if the given key is in the tree.
func (t *BTree) Has(key Item) bool {
	return (*BTreeG[Item])(t).Has(key)
}

// ReplaceOrInsert 方法向树中添加给定的项。
// 如果树中已有一个等于给定项的元素，它将被移除并返回。
// 否则，返回 nil。
//
// 不能向树中添加 nil（会引发 panic）。
func (t *BTree) ReplaceOrInsert(item Item) Item {
	// 调用泛型 B-树的 ReplaceOrInsert 方法
	i, _ := (*BTreeG[Item])(t).ReplaceOrInsert(item)
	return i // 返回被替换或插入的项
}

// AscendRange 为树中范围 [greaterOrEqual, lessThan) 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTree) AscendRange(greaterOrEqual, lessThan Item, iterator ItemIterator) {
	// 调用泛型 B-树的 AscendRange 方法
	(*BTreeG[Item])(t).AscendRange(greaterOrEqual, lessThan, (ItemIteratorG[Item])(iterator))
}

// AscendLessThan 为树中范围 [first, pivot) 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTree) AscendLessThan(pivot Item, iterator ItemIterator) {
	// 调用泛型 B-树的 AscendLessThan 方法
	(*BTreeG[Item])(t).AscendLessThan(pivot, (ItemIteratorG[Item])(iterator))
}

// AscendGreaterOrEqual 为树中范围 [pivot, last] 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTree) AscendGreaterOrEqual(pivot Item, iterator ItemIterator) {
	// 调用泛型 B-树的 AscendGreaterOrEqual 方法
	(*BTreeG[Item])(t).AscendGreaterOrEqual(pivot, (ItemIteratorG[Item])(iterator))
}

// Ascend 为树中范围 [first, last] 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTree) Ascend(iterator ItemIterator) {
	// 调用泛型 B-树的 Ascend 方法
	(*BTreeG[Item])(t).Ascend((ItemIteratorG[Item])(iterator))
}

// DescendRange 为树中范围 [lessOrEqual, greaterThan) 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTree) DescendRange(lessOrEqual, greaterThan Item, iterator ItemIterator) {
	// 调用泛型 B-树的 DescendRange 方法
	(*BTreeG[Item])(t).DescendRange(lessOrEqual, greaterThan, (ItemIteratorG[Item])(iterator))
}

// DescendLessOrEqual 为树中范围 [pivot, first] 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTree) DescendLessOrEqual(pivot Item, iterator ItemIterator) {
	// 调用泛型 B-树的 DescendLessOrEqual 方法
	(*BTreeG[Item])(t).DescendLessOrEqual(pivot, (ItemIteratorG[Item])(iterator))
}

// DescendGreaterThan 为树中范围 [last, pivot) 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTree) DescendGreaterThan(pivot Item, iterator ItemIterator) {
	// 调用泛型 B-树的 DescendGreaterThan 方法
	(*BTreeG[Item])(t).DescendGreaterThan(pivot, (ItemIteratorG[Item])(iterator))
}

// Descend 为树中范围 [last, first] 内的每个值调用迭代器，直到迭代器返回 false。
func (t *BTree) Descend(iterator ItemIterator) {
	// 调用泛型 B-树的 Descend 方法
	(*BTreeG[Item])(t).Descend((ItemIteratorG[Item])(iterator))
}

// Len 返回树中当前的项数。
func (t *BTree) Len() int {
	// 调用泛型 B-树的 Len 方法并返回结果
	return (*BTreeG[Item])(t).Len()
}

// Clear 移除 btree 中的所有项。
// 如果 addNodesToFreelist 为 true，则 t 的节点作为此调用的一部分被添加到它的 freelist 中，
// 直到 freelist 被填满。否则，根节点简单地取消引用，子树留给 Go 的常规垃圾回收过程。
//
// 这比对所有元素调用 Delete 快得多，因为 Delete 需要在树中找到/移除
// 每个元素，并相应地更新树。它也比创建新树来替换旧树稍微快一些，
// 因为旧树的节点被回收到 freelist 中以供新树使用，而不是丢给垃圾回收器。
//
// 这个调用的时间复杂度：
//
//	O(1): 当 addNodesToFreelist 为 false 时，这是一个单一操作。
//	O(1): 当 freelist 已经满了，它会立即退出。
//	O(freelist size): 当 freelist 为空且所有节点都由这棵树拥有时，
//	   节点会被添加到 freelist 中，直到它被填满。
//	O(tree size): 当所有节点都由另一棵树拥有时，会迭代所有节点以寻找可添加到 freelist 的节点，
//	   由于所有权问题，没有节点会被添加。
func (t *BTree) Clear(addNodesToFreelist bool) {
	// 调用泛型 B-树的 Clear 方法
	(*BTreeG[Item])(t).Clear(addNodesToFreelist)
}
