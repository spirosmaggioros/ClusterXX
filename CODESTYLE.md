# Classes
  - Classes must always exist inside a clusterxx namespace
  - Private data types should come first
  - Private parameters and functions should start with two underscores '__*'
  - Private parameters are initialized in the constructor, we do otherwise only in special cases
  - Use assertions, no throw-catch exceptions
  - For each class, there should be two .hpp files, one for the class declaration and one for the class implementation
  - Don't forget to include the implementation header in the end of the declaration header
  - All of our methods follows sklearn's API, so make sure you follow it too

# Types
  - For data, we use armadillo's mat or vec
  - The above insists that data is in double
  - Note that we use armadillo where is needed, sometimes we return std::vector types to make user's life easier

# Generic
  - Function implementations shouldn't be more than 40 lines of code. If so, make sure you use helper functions or just make your implementation better
  - Make sure you write some unit tests for every line of code you add, we don't care about quantity but quality of unit tests
  - Use brackets for if, for, while statements. Even if there's one line of code inside, and never in one line
  - We configured clang for this project, you can make sure that you follow our style format by clang formatting your file(using clang-format)
