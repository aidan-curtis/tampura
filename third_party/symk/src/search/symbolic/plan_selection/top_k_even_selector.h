#ifndef SYMBOLIC_TOP_K_EVEN_SELECTOR_H
#define SYMBOLIC_TOP_K_EVEN_SELECTOR_H

#include "plan_selector.h"

namespace symbolic {
class TopKEvenSelector : public PlanSelector {
public:
    TopKEvenSelector(const options::Options &opts);

    ~TopKEvenSelector() {}

    void add_plan(const Plan &plan) override;

    std::string tag() const override {return "Top-K (even plan length)";}
};
}

#endif
