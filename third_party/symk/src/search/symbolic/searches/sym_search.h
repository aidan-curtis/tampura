#ifndef SYMBOLIC_SYM_SEARCH_H
#define SYMBOLIC_SYM_SEARCH_H

#include "../sym_estimate.h"
#include "../sym_params_search.h"
#include "../sym_state_space_manager.h"
#include "../sym_utils.h"
#include <map>
#include <memory>
#include <vector>

namespace symbolic {
class SymbolicSearch;

class SymSearch {
protected:
    // Attributes that characterize the search:
    std::shared_ptr<SymStateSpaceManager>
    mgr;   // Symbolic manager to perform bdd operations
    SymParamsSearch p;

    SymbolicSearch *engine; // Access to the bound and notification of new
                            // solutions

public:
    SymSearch(SymbolicSearch *eng, const SymParamsSearch &params);

    virtual ~SymSearch() {}

    SymStateSpaceManager *getStateSpace() {return mgr.get();}

    std::shared_ptr<SymStateSpaceManager> getStateSpaceShared() const {
        return mgr;
    }

    virtual void step() = 0;
    virtual std::string get_last_dir() const = 0;

    virtual void stepImage(int maxTime, int maxNodes) = 0;

    virtual int getF() const = 0;
    virtual bool finished() const = 0;
};
}
#endif
