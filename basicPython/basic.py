# coding:utf-8
# 1 数据结构

# 列表
#
# 列表是处理一组有序 项目的数据结构，即你可以在一个列表中存储一个序列的项目。
# 并且里面的值是能够被改变的;列表中的项目应该包括在放括号中，这样Python就知道你是在指明一个列表。
# 一旦创建列表，你可以添加、删除、或是搜索列表中的项目。
# 由于可以增加或删除项目，我们说列表是可变的数据类型，即这种类型是可以被改变的。

shoplist = ['apple', 'mango', 'carrot', 'banana']
print shoplist

shoplist.sort()
print 'The first item i will buy is', shoplist[0]

del shoplist[3]
print '删除某一个元素: ', shoplist

# 元祖
# 元祖和列表十分类似，只不过元祖和字符串一样是不可变的，即你不能修改元祖。
# 元祖通过圆括号中用逗号分割的项目定义。
# 元祖通常用在使语句或用户定义的函数能够完全的采用一组值的时候，即被使用的元祖的值不会改变
zoo = ('wolf', 'elephant', 'penguin')
print 'Number of animals in the zoo is', len(zoo)

new_zoo = ('monkey', 'dolphin', zoo)
print 'Number of animals in the new zoo is', len(new_zoo)
print 'All animals in new zoo are', new_zoo

print 'Animals brought from old zoo are', new_zoo[2]
print 'Last animal brought from old zoo is', new_zoo[2][2]
