#pragma once

namespace tds {
enum SerializationMode { SERIALIZE_DYNAMICS = 1, SERIALIZE_GEOMETRY = 2 };

inline SerializationMode operator|(SerializationMode a, SerializationMode b) {
  return static_cast<SerializationMode>(static_cast<int>(a) |
                                        static_cast<int>(b));
}

template <typename Algebra>
struct Serializable {
  using Scalar = typename Algebra::Scalar;
  using Iter = typename std::vector<Scalar>::iterator;
  using ConstIter = typename std::vector<Scalar>::const_iterator;

  virtual void set_serializable(bool s = true) { is_serializable_ = s; }
  virtual bool is_serializable() const { return is_serializable_; }

  virtual size_t serialization_size(SerializationMode mode) const {
    if (!is_serializable_) {
      return 0;
    }
    return serialization_size_(mode);
  }
  virtual void serialize(Iter output, SerializationMode mode) const {
    if (!is_serializable_) {
      return;
    }
    serialize_(output, mode);
  }
  virtual void deserialize(ConstIter input, SerializationMode mode) {
    if (!is_serializable_) {
      return;
    }
    deserialize_(input, mode);
  }

 protected:
  virtual size_t serialization_size_(SerializationMode mode) const = 0;
  virtual void serialize_(Iter output, SerializationMode mode) const = 0;
  virtual void deserialize_(ConstIter input, SerializationMode mode) = 0;

  bool is_serializable_{false};
};
}  // namespace tds