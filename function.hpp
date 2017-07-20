#ifndef FUNCTION_HPP
#define FUNCTION_HPP

// a function has an input and output dimension and implements fwd and bwd pass
struct Function {
    // every layer has to implement forward and backward
    virtual void forward(const Tensor& input, Tensor& output) = 0;
    virtual void init_forward(const Tensor&, Tensor&) {};
    virtual void backward(const Tensor& doutput, Tensor& dinput) = 0;
    virtual void init_backward(const Tensor&, Tensor&) {};

    // return the input dimensions
    virtual const TensorDesc& getInputDesc() const = 0;

    /// returns the output dimensions
    virtual const TensorDesc& getOutputDesc() const = 0;

    // Prints the input and output dimensions to the given stream
    std::ostream& write_dims(std::ostream& os) const {
        return os << getInputDesc() << "->" << getOutputDesc();
    }

    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "Function";
    }

    virtual std::ostream& write(std::ostream& os) const {
        return this->write_dims(this->write_name(os) << " ");
    }
};

/* a Layer is a Function for which the input and output dimensions are known
 * at construction time and it buffers these
 */
struct Layer : public Function {
    TensorDesc input_desc;
    TensorDesc output_desc;

    Layer(const Dim& input_desc, const Dim& output_desc)
        : input_desc(input_desc), output_desc(output_desc) {}

    Layer(const TensorDesc& input_desc, const TensorDesc& output_desc)
        : input_desc(input_desc), output_desc(output_desc) {}

    virtual const TensorDesc& getInputDesc() const override {
        return input_desc;
    }

    virtual const TensorDesc& getOutputDesc() const override {
        return output_desc;
    }

    virtual std::ostream& write_name(std::ostream& os) const override {
        return os << "Layer";
    }
};

std::ostream& operator<<(std::ostream& os, const Function& l) {
    return l.write(os);
}

/*
struct Model {

    void init_forward();
    void init_backward();

    void forward();
    void backward();

    std::string get_name() const;
};
*/

#endif // FUNCTION_HPP
