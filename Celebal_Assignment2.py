class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=' -> ' if current.next else '\n')
            current = current.next

    def delete_nth_node(self, n):
        if not self.head:
            raise Exception("Cannot delete from an empty list")
        if n <= 0:
            raise Exception("Index must be a positive integer")
        if n == 1:
            self.head = self.head.next
            return
        current = self.head
        count = 1
        while current and count < n - 1:
            current = current.next
            count += 1
        if not current or not current.next:
            raise Exception("Index out of range")
        current.next = current.next.next

linked_list = LinkedList()
linked_list.add_node(10)
linked_list.add_node(20)
linked_list.add_node(30)
linked_list.add_node(40)
print("Original list:")
linked_list.print_list()
linked_list.delete_nth_node(3)
print("After deleting 3rd node:")
linked_list.print_list()

try:
    linked_list.delete_nth_node(10)
except Exception as e:
    print("Error:", e)

try:
    empty_list = LinkedList()
    empty_list.delete_nth_node(1)
except Exception as e:
    print("Error:", e)
