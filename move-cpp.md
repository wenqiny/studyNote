# A case for move
## TODO: see asm code later:

```
#include <iostream>
#include <string>

// Simplified std::remove_reference implementation
template <typename T>
struct remove_reference {
    using type = T; 
};

// Specialization for lvalue references (e.g., int&)
template <typename T>
struct remove_reference<T&> {
    using type = T;
};

// Specialization for rvalue references (e.g., int&&)
template <typename T>
struct remove_reference<T&&> {
    using type = T;
};


// Simplified std::move implementation
template <typename T>
typename remove_reference<T>::type&& my_move(T&& arg) {
    // This function casts its argument into an rvalue reference,
    // signaling that the object can be moved from.
    return static_cast<typename remove_reference<T>::type&&>(arg);
}

class A {
public:
    A() = default;  // Default constructor

    A(const A& other) {  // Copy constructor (for comparison)
        std::cout << "Copy constructor called\n";
        data = other.data;
    }

    A(A&& other) noexcept { // Move constructor
        std::cout << "Move constructor called\n";
        data = my_move(other.data); // Use std::move or our simplified move
        other.data = "";  // Reset the source object
    }
    
    std::string data; 
};

int main() {
    A a1;
    a1.data = "Hello";
    std::cout << "a1 before move: " << a1.data << std::endl;
    A a2 = my_move(a1);       
    std::cout << "a1 after move: " << a1.data << std::endl;
    std::cout << "a2: " << a2.data << std::endl;

    return 0;
}
```