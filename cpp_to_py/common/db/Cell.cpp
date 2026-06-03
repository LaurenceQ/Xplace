#include "Cell.h"

#include <cstring>

#include "DatabaseClass.h"
#include "Geometry.h"
#include "Pin.h"
#include "common/lib/Liberty.h"

using namespace db;

/***** Cell *****/

Cell::~Cell() {
    for (Pin* pin : _pins) {
        delete pin;
    }
    _pins.clear();
}

Pin* Cell::pin(const char* name) const {
    if (name == nullptr) {
        return nullptr;
    }
    return pin(name, name + std::strlen(name));
}

Pin* Cell::pin(const char* begin, const char* end) const {
    if (!_type || begin == nullptr || end == nullptr || begin >= end) {
        return nullptr;
    }

    const char first = *begin;
    const std::size_t name_size = static_cast<std::size_t>(end - begin);
    const std::vector<PinType*>& type_pins = _type->pins;
    for (unsigned i = 0; i != type_pins.size(); ++i) {
        const std::string& pin_name = type_pins[i]->name();
        if (pin_name.size() == name_size &&
            !pin_name.empty() &&
            pin_name[0] == first &&
            std::memcmp(pin_name.data(), begin, name_size) == 0) {
            return _pins[i];
        }
    }
    return nullptr;
}

Pin* Cell::pin(const string& name) const {
    return pin(name.data(), name.data() + name.size());
}

void Cell::ctype(CellType* t) {
    if (!t) {
        return;
    }
    if (_type) {
        logger.error("type of cell %s already set", _name.c_str());
        return;
    }
    _type = t;
    _type->usedCount.fetch_add(1, std::memory_order_relaxed);
    _pins.resize(_type->pins.size(), nullptr);
    for (unsigned i = 0; i != _pins.size(); ++i) {
        _pins[i] = new Pin(this, i);
    }
}

int Cell::lx() const { return _lx; }
int Cell::ly() const { return _ly; }
int Cell::orient() const { return _orient; }

bool Cell::placed() const { return (lx() != INT_MIN) && (ly() != INT_MIN); }
// int Cell::siteWidth() const { return width() / database.siteW; }
// int Cell::siteHeight() const { return height() / database.siteH; }

void Cell::place(int x, int y) {
    if (_fixed) {
        logger.warning("moving fixed cell %s to (%d,%d)", _name.c_str(), x, y);
    }
    _lx = x;
    _ly = y;
}

void Cell::place(int x, int y, int orient) {
    if (_fixed) {
        logger.warning("moving fixed cell %s to (%d,%d)", _name.c_str(), x, y);
    }
    _lx = x;
    _ly = y;
    _orient = orient;
}

void Cell::unplace() {
    if (_fixed) {
        logger.warning("unplace fixed cell %s", _name.c_str());
    }
    _lx = _ly = INT_MIN;
    _orient = -1;
}

/***** Cell Type *****/

CellType::~CellType() {
    for (PinType* pin : pins) {
        delete pin;
    }
}

PinType* CellType::addPin(const string& name, const char direction, const char type) {
    PinType* newpintype = new PinType(name, direction, type);
    pins.push_back(newpintype);
    return newpintype;
}

PinType* CellType::getPin(string& name) {
    for (int i = 0; i < (int)pins.size(); i++) {
        if (pins[i]->name() == name) {
            return pins[i];
        }
    }
    return nullptr;
}

void CellType::setOrigin(int x, int y) {
    _originX = x;
    _originY = y;
}

bool CellType::operator==(const CellType& r) const {
    if (width != r.width || height != r.height) {
        return false;
    } else if (_originX != r.originX() || _originY != r.originY() || _symmetry != r.symmetry() ||
               pins.size() != r.pins.size()) {
        return false;
    } else if (edgetypeL != r.edgetypeL || edgetypeR != r.edgetypeR) {
        return false;
    } else {
        //  return PinType::comparePin(pins, r.pins);
        for (unsigned i = 0; i != pins.size(); ++i) {
            if (*pins[i] != *r.pins[i]) {
                return false;
            }
        }
    }
    return true;
}
