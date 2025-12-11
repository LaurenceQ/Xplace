

#include "GTDatabase.h"

#include "common/common.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Pin.h"
#include "common/lib/Liberty.h"
#include "common/lib/Timing.h"
#include "common/lib/sdc/sdc.h"
#include "io_parser/gp/GPDatabase.h"

namespace gt {

bool GTDatabase::is_redundant_timing(const TimingArc* timing_arc, Split el) {
    if (timing_arc->from_port_->name == timing_arc->to_port_->name) return true;
    if (timing_arc->related_port_name_.empty()) return true;
    if (timing_arc->timing_type_ == TimingType::non_seq_setup_rising || timing_arc->timing_type_ == TimingType::non_seq_setup_falling || timing_arc->timing_type_ == TimingType::non_seq_hold_rising ||
        timing_arc->timing_type_ == TimingType::non_seq_hold_falling || timing_arc->timing_type_ == TimingType::clear || timing_arc->timing_type_ == TimingType::preset || 
        timing_arc->timing_type_ == TimingType::removal_falling || timing_arc->timing_type_ == TimingType::removal_rising ||
        timing_arc->timing_type_ == TimingType::recovery_falling || timing_arc->timing_type_ == TimingType::recovery_rising)
        return true;
    switch (el) {
        case MIN:
            if (timing_arc->is_max_constraint()) {
                return true;
            }
            break;
        case MAX:   
            if (timing_arc->is_min_constraint()) {
                return true;
            }
            break;
    }
    return false;
}
string debus(string name) {
    // remove the bus bit index at last
    // A[0] -> A
    if (name.back() == ']') {
        auto pos = name.find_last_of('[');
        if (pos != string::npos) {
            return name.substr(0, pos);
        }
    } else 
        return name;
}

void GTDatabase::gate_sizing_init() {
    unordered_map<size_t, int> type_hash2equivalent_cell_id_map;
    cell_area.resize(rawdb.celltypes.size());
    lib_cell2equivalen_cell_map.resize(rawdb.celltypes.size());
    for (const auto& pair : cell_libs_[MIN]->lib_cells_hash_map_) {
        type_hash2equivalent_cell_id_map.emplace(pair.first, (int)type_hash2equivalent_cell_id_map.size());
        equivalent_cell_list_end.emplace_back(equivalent_cell_list_end.back() + pair.second.size());
        equivalent_cell_num_pin.emplace_back(rawdb.celltypes[pair.second[0]->cell_type_->libcell()]->liberty_cell->ports_.size());
        vector<LibertyCell*> cell_type_list = pair.second;
        std::sort(cell_type_list.begin(), cell_type_list.end(), [](const LibertyCell* a, const LibertyCell* b) {
            return a->area_ < b->area_;
        });
        printf("===========\n");
        for(LibertyCell* cell_type : cell_type_list) {
            printf("cell_type:%s area:%.5f cell_id:%d\n", cell_type->name.c_str(), *(cell_type -> area_), cell_type -> cell_type_ -> libcell());
            lib_cell2equivalen_cell_map[cell_type->cell_type_->libcell()] = equivalent_cell_list.size();
            equivalent_cell_list.emplace_back(cell_type->cell_type_->libcell());

            if(cell_type -> area_ == std::nullopt)
                cell_area[cell_type->cell_type_->libcell()] = 0;
            else 
                cell_area[cell_type->cell_type_->libcell()] = *(cell_type -> area_);
        }
    }
    for(int pin_id = 0; pin_id < gpdb.getPins().size(); pin_id++){
        pin_IO_direction.emplace_back(gpdb.getPins()[pin_id].getDirection());
        int libcell_id = pin_id2cell_type_id[pin_id];
        if(libcell_id == -1){
            pin_id2equivalent_cell_id.emplace_back(-1); // PI or PO
            continue;
        }
        size_t type_hash = rawdb.celltypes[libcell_id]->liberty_cell->size_hash;
        auto it = type_hash2equivalent_cell_id_map.find(type_hash);
        if (it != type_hash2equivalent_cell_id_map.end()) {
            pin_id2equivalent_cell_id.emplace_back(it->second);
        } else {
            logger.error("GTDatabase::rehash: type_hash %llu not found in rehash_map", type_hash);
        }
    }
    for(int arc_id = 0; arc_id < num_arcs; arc_id++){
        if(arc_types[arc_id] == 1){
            arc_lambda.emplace_back(0.1);
            arc_lambda.emplace_back(0.1);
        }
        else {
            arc_lambda.emplace_back(0);
            arc_lambda.emplace_back(0);            
        }
    }
    for(int po_id = 0; po_id < num_POs; po_id++){
        po_lambda.emplace_back(0.1f);
        po_lambda.emplace_back(0.1f);
    }
    // int cnt = 0;
    // for(int i = 0;i < gpdb.getNodes().size(); i++){
    //     // std::cout << i << "th Cell is " << gpdb.getNodes()[i].getName()<<std::endl;
    //     std::map<int, int> port_offset2pin_id;
    //     port_offset2pin_id.clear();
    //     for(int pin_id : gpdb.getNodes()[i].pins()){
    //         int port_offset = pin_id2port_offset_id[pin_id];
    //         // assert(port_offset2pin_id.find(port_offset) == port_offset2pin_id.end());
    //         if(port_offset2pin_id.find(port_offset) != port_offset2pin_id.end())cnt++;
    //         port_offset2pin_id[port_offset] = pin_id;
    //         // std::cout<<"Pin " << gpdb.getPins()[pin_id].getName() << "# " << pin_id << " is at "  << port_offset << "th port" << std::endl;
    //     }
    // }
    // std::cout<< cnt << "pins have the same port offset" << std::endl;
}
void GTDatabase::swap_gate_type(int node_id, int type_index) {
    int libcell_id = rawdb.cells[gpdb.getNodes()[node_id].getOriDBId()]->ctype()->libcell();
    size_t type_hash = rawdb.celltypes[libcell_id]->liberty_cell->size_hash;
    vector<LibertyCell*>& liberty_cells = rawdb.cell_libs_[MIN]->lib_cells_hash_map_[type_hash];
    if (type_index >= liberty_cells.size()) {
        logger.error("GTDatabase::swap_gate_type: type_index %d is out of range for type_hash %llu with size %d", type_index, type_hash, liberty_cells.size());
        return;
    }
    int change_libcell_id = liberty_cells[type_index]->cell_type_->libcell();

    for_each_el(el) {
        for (int pin_id : gpdb.getNodes()[node_id].pins()) {
            int pin_id2port_start = liberty_cell_type2port_list_end[change_libcell_id]; // liberty cell type -> port#. type1:3ports, type2:4ports. 0->0 1->3 2->7
            int pin_id2port_offset = pin_id2port_offset_id[pin_id]; 
            int port_id = pin_id2port_start + pin_id2port_offset;
            std::cout << "swap gate type for node " << node_id << " name:" <<
                    rawdb.celltypes[libcell_id]->name <<":"<<
                    rawdb.celltypes[libcell_id]->liberty_cell->ports_[pin_id2port_offset]->name << " to " <<
                    rawdb.celltypes[change_libcell_id]->name << ":" <<
                    rawdb.celltypes[change_libcell_id]->liberty_cell->ports_[pin_id2port_offset]->name << "\n";
            int start = liberty_port2timing_list_end[2 * port_id + el]; // port# -> liberty timing list#
            int timing_id_start = pin_id2timing_arc_list_start[2 * pin_id + el]; // pin# -> timing graph arc#
            int timing_id_end = pin_id2timing_arc_list_end[2 * pin_id + el];
            for (int i = timing_id_start; i < timing_id_end; i++) {
                if (timing_arc_id_map[i] == -1) continue;  // skip if timing arc is not found
                int arc_id_non_el = i / 2;
                int in_port_id = timing_arc_in_port_id[arc_id_non_el]; // graph timing arc -> port_lib_timing_arc_offset
                std::cout << "timing arc id " << 
                timing_arc_id_map[i] << " -> " <<
                start + in_port_id << "\n";
                timing_arc_id_map[i] = start + in_port_id;
                timing_raw_db.timing_arc_id_map[i] = start + in_port_id;
            }
        }
    }
}
void GTDatabase::change_cell_size(int output_pin_id, unsigned libcell){
    int node_id = gpdb.getPins()[output_pin_id].getParNodeId();
    int cell_id = gpdb.getNodes()[node_id].getOriDBId();
    if(rawdb.cells[cell_id] -> ctype() -> libcell() == libcell)return ;
    printf("output_pin_id:%d ", output_pin_id);
    rawdb.changeCellSizing(cell_id, libcell);
    gpdb.changeCellSizing(node_id, cell_id);
}
vector<index_type> GTDatabase::get_driver_gate_sink_arc(int node_id) {
    vector<index_type> driver_gate_sink_arcs;
    std::cout << "get_driver_gate_sink_arc for cell " << gpdb.getNodes()[node_id].getName() << "\n";
    for (int pin_id : gpdb.getNodes()[node_id].pins()) {
        int net_id = gpdb.getPins()[pin_id].getParNetId();
        int rpin_id = gpdb.getNets()[net_id].pins()[0];
        if(pin_id != rpin_id){ // driver pin arcs
            if(gpdb.getPins()[rpin_id].getDirection() != 'o')continue;
            std::cout << "get driver pin arcs for pin " << gpdb.getPins()[pin_id].getName() << " drived by net " << gpdb.getNets()[net_id].getName() << " and root pin " << gpdb.getPins()[rpin_id].getName() << "\n";
            for(index_type i = pin_backward_arc_list_end[rpin_id]; i < pin_backward_arc_list_end[rpin_id+1]; i++) {
                index_type arc_id = pin_backward_arc_list[i];
                index_type fpin_id = timing_arc_from_pin_id[arc_id];
                std::cout << "arc_id " << arc_id << " from pin " << gpdb.getPins()[fpin_id].getName() << "\n";
                if (arc_id == -1) continue;  // skip if arc is not found
                assert(arc_types[arc_id] == 1);
                driver_gate_sink_arcs.push_back(arc_id);
            }
        }
        else { // sink pin
            std::cout << "get sink pin arcs for pin " << gpdb.getPins()[pin_id].getName() << " driving net " << gpdb.getNets()[net_id].getName() << "\n";
            for(auto spin : gpdb.getNets()[net_id].pins()) { // sink pin arcs
                if (spin == pin_id) continue; // skip driver pin
                std::cout << "cur pin driving sink pin " << gpdb.getPins()[spin].getName() << "\n";
                for(index_type i = pin_forward_arc_list_end[spin]; i < pin_forward_arc_list_end[spin+1]; i++) {
                    index_type arc_id = pin_forward_arc_list[i];
                    index_type tpin_id = timing_arc_to_pin_id[arc_id];
                    std::cout << "arc_id " << pin_forward_arc_list[i] << " to pin " << gpdb.getPins()[tpin_id].getName() << "\n";
                    if (arc_id == -1) continue;  // skip if arc is not found
                    assert(arc_types[arc_id] == 1);
                    driver_gate_sink_arcs.push_back(arc_id);
                }
            }
            std::cout << "get gate arcs for pin " << gpdb.getPins()[pin_id].getName() << "\n";
            for(index_type i = pin_backward_arc_list_end[pin_id]; i < pin_backward_arc_list_end[pin_id+1]; i++) { //gate arcs
                index_type arc_id = pin_backward_arc_list[i];
                index_type fpin_id = timing_arc_from_pin_id[arc_id];
                std::cout << "arc_id " << arc_id << " to pin " << gpdb.getPins()[fpin_id].getName() << "\n";
                if (arc_id == -1) continue;  // skip if arc is not found
                assert(arc_types[arc_id] == 1);
                driver_gate_sink_arcs.push_back(arc_id);
            }
        }
    }
    return driver_gate_sink_arcs;
}
GTDatabase::GTDatabase(shared_ptr<db::Database> rawdb_, shared_ptr<gp::GPDatabase> gpdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_) : rawdb(*rawdb_), gpdb(*gpdb_), timing_raw_db(*timing_raw_db_) {
    cell_libs_[MIN] = rawdb.cell_libs_[MIN];
    cell_libs_[MAX] = rawdb.cell_libs_[MAX];
}


set<string> common_timing(map<string, TimingArc *> a, map<string, TimingArc *> b) {
    set<string> common;
    for (auto& [k, v] : a) {
        if (b.find(k) != b.end()) {
            common.insert(k);
        }
    }
    return common;
}

void GTDatabase::ExtractTimingGraph() {
    res_unit = cell_libs_[MIN]->resistance_unit_->value();
    cap_unit = cell_libs_[MIN]->capacitance_unit_->value();
    time_unit = cell_libs_[MIN]->time_unit_->value();
    pin_names = gpdb.getPinNames();
    net_names = gpdb.getNetNames();
    
    //  Flatten Liberty Cell Timing
    for (db::CellType* cell_type : rawdb.celltypes) {
        string cell_type_name = cell_type->name;
        array<LibertyCell*, 2> liberty_cell_view = {cell_libs_[MIN]->get_cell(cell_type_name), cell_libs_[MAX]->get_cell(cell_type_name)};
        if (!liberty_cell_view[MIN] || !liberty_cell_view[MAX]) {
            liberty_cell_type2port_list_end.push_back(liberty_cell_type2port_list_end.back());
            continue;
        }
        liberty_cell_type2port_list_end.push_back(liberty_cell_type2port_list_end.back() + liberty_cell_view[MIN]->ports_.size());
        for (int i = 0; i < liberty_cell_view[MIN]->ports_.size(); i++) {
            array<LibertyPort*, 2> liberty_port_view = {liberty_cell_view[MIN]->ports_[i], liberty_cell_view[MAX]->ports_[i]};
            for_each_el(el) {
                for(int rf = 0; rf < 2; rf++){
                    // float lib_pin_cap = liberty_port_view[el]->port_capacitance_[rf].value_or(nanf(""));
                    // if(isnan(lib_pin_cap)){
                    float lib_pin_cap = liberty_port_view[el]->port_capacitances_[rf][el].value_or(nanf(""));
                    // }
                    if(isnan(lib_pin_cap)){
                        lib_pin_cap = liberty_port_view[el^1]->port_capacitances_[rf][el].value_or(nanf(""));
                    }
                    liberty_port_capacitance.emplace_back(lib_pin_cap);
                }
                liberty_port_capacitance.emplace_back(liberty_port_view[el]->port_capacitance_[2].value_or(0.0f));
            }
            for_each_el(el) {
                liberty_port2timing_list_end.push_back(liberty_port2timing_list_end.back() + liberty_port_view[el]->timing_arcs_non_cond_non_bundle_.size());
                for (int j = 0; j < liberty_port_view[el]->timing_arcs_non_cond_non_bundle_.size(); j++) {
                    liberty_timing_arcs.push_back(liberty_port_view[el]->timing_arcs_non_cond_non_bundle_[j]);
                }
            }
        }
    }

    //  Traverse Circuit Pins
    //
    num_pins = gpdb.getPins().size();
    pin_names = gpdb.getPinNames();
    net_names = gpdb.getNetNames();
    pin_id2cell_type_id.resize(num_pins);
    pin_id2port_offset_id.resize(num_pins);
    STA_pins.resize(num_pins, nullptr);
    pin_capacitance.resize(2 * 3 * num_pins, 0.0f);
    for (auto& gppin : gpdb.getPins()) {
        int pin_id = gppin.getId();
        string pin_name = gppin.getName();
        string pin_macro_name = gppin.getMacroName();
        STA_pins[pin_id] = new STAPin();
        auto [ori_node_id, ori_node_pin_id, ori_net_id] = gppin.getOriDBInfo();
        if (ori_node_pin_id == -1) {
            auto dbiopin = rawdb.iopins[ori_node_id];
            pin_id2cell_type_id[pin_id] = -1;
            if (dbiopin->type->direction() == 'i') {
                primary_outputs.push_back(pin_id);
                endpoints_id.push_back(pin_id);
                primary_output2pin_id[pin_name] = pin_id;
            } else if (dbiopin->type->direction() == 'o') {
                primary_inputs.push_back(pin_id);
                primary_input2pin_id[pin_name] = pin_id;
                // std::cout<<"primary input pin " << pin_name << " id " << pin_id << "\n";
            }
        } else {
            auto& dbcell = rawdb.cells[ori_node_id];
            LibertyCell* liberty_cell = dbcell->ctype()->liberty_cell;
            pin_id2cell_type_id[pin_id] = dbcell->ctype()->libcell();
            if (dbcell->ctype()->cls == "BLOCK") pin_macro_name = debus(pin_macro_name);
            pin_id2port_offset_id[pin_id] = liberty_cell->ports_map_[pin_macro_name];

            int liberty_port_id = liberty_cell_type2port_list_end[pin_id2cell_type_id[pin_id]] + pin_id2port_offset_id[pin_id]; //many port_offset_id = 0
            

            for_each_el(el) {
                pin_capacitance[6 * pin_id + el * 2 + 0] = liberty_port_capacitance[6 * liberty_port_id + el * 3 + 0];
                pin_capacitance[6 * pin_id + el * 2 + 1] = liberty_port_capacitance[6 * liberty_port_id + el * 3 + 1];
                pin_capacitance[6 * pin_id + 4 + el] = liberty_port_capacitance[6 * liberty_port_id + el * 3 + 2];
            }
        }
    }


    //  Map Pin to Liberty Timing
    //
    auto connect_from_to_pin = [&](int from_pin_id, int to_pin_id) -> pair<STAPin*, STAPin*> {
        STAPin* from_pin = STA_pins[from_pin_id];
        STAPin* to_pin = STA_pins[to_pin_id];
        from_pin->fanout_pin_ids.insert(to_pin_id);
        to_pin->fanin_pin_ids.insert(from_pin_id);
        timing_arc_from_pin_id.push_back(from_pin_id);
        timing_arc_to_pin_id.push_back(to_pin_id);
        from_pin->timing_arc_out.push_back(num_arcs);
        to_pin->timing_arc_in.push_back(num_arcs);
        return {from_pin, to_pin};
    };

    for (auto& gpnet : gpdb.getNets()) {
        int driver_pin_id = gpnet.pins()[0];
        if(gpnet.getName() == "VDD")assert(gpdb.getPins()[gpnet.pins()[0]].getDirection() == 'o');
        for (index_type i = 1; i < static_cast<index_type>(gpnet.pins().size()); i++) {
            int sink_pin_id = gpnet.pins()[i];
            auto [from_pin, to_pin] = connect_from_to_pin(driver_pin_id, sink_pin_id);
            timing_arc_id_map.push_back(-1);
            timing_arc_id_map.push_back(-1);
            timing_arc_in_port_id.push_back(-1);
            arc_types.push_back(0);
            arc_id2test_id.push_back(-1);
            num_arcs++;
        }
    }

    cell_node_type_map.resize(gpdb.getNodes().size(), -1);
    pin_id2timing_arc_list_start.resize(2 * num_pins, 0);
    pin_id2timing_arc_list_end.resize(2 * num_pins, 0);
    is_FF_Q.resize(num_pins, 0);
    for (auto& dbcell : rawdb.cells) {
        int gpdb_id = dbcell->gpdb_id;
        int libcell_id = dbcell->ctype()->libcell();
        cell_node_type_map[gpdb_id] = libcell_id;
        for_each_el(el) {
            for (int pin_id : gpdb.getNodes()[gpdb_id].pins()) {
                pin_id2timing_arc_list_start[2 * pin_id + el] = timing_arc_id_map.size();
                int pin_id2port_start = liberty_cell_type2port_list_end[libcell_id];
                int pin_id2port_offset = pin_id2port_offset_id[pin_id];
                int port_id = pin_id2port_start + pin_id2port_offset;
                int start = liberty_port2timing_list_end[2 * port_id + el];
                int end = liberty_port2timing_list_end[2 * port_id + el + 1];
                bool is_endpoint = false;
                for (int i = start; i < end; i++) {
                    TimingArc* timing_arc = liberty_timing_arcs[i];
                    if (is_redundant_timing(timing_arc, el)) {
                        continue;
                    }
                    array<int, 2> timing_view = {-1, -1};
                    timing_view[el] = i;

                    int from_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->from_port_->name);
                    int to_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->to_port_->name);
                    if (dbcell->ctype()->cls == "BLOCK") {
                        assert(!(from_pin_id == -1 && to_pin_id == -1));
                        if (from_pin_id == -1) from_pin_id = pin_id; // connect to the pin itself
                        if (to_pin_id == -1) to_pin_id = pin_id;     //
                    } else if(from_pin_id == -1 || to_pin_id == -1)continue;
                    auto [from_pin, to_pin] = connect_from_to_pin(from_pin_id, to_pin_id);
                    timing_arc_id_map.push_back(timing_view[MIN]);
                    timing_arc_id_map.push_back(timing_view[MAX]);
                    timing_arc_in_port_id.push_back(i - start);
                    arc_types.push_back(1);
                    num_arcs++;

                    if (timing_arc->is_constraint()) {
                        
                        is_endpoint = to_pin_id == pin_id;
                        arc_id2test_id.push_back(num_tests++);
                        test_id2_arc_id.push_back(num_arcs - 1);
                        // endpoints_id.push_back(to_pin_id);
                    } else {
                        arc_id2test_id.push_back(-1);
                        if(gpdb.getPins()[from_pin_id].getType() == 'c'){
                            is_FF_Q[to_pin_id] = 1;
                        }
                    }
                }
                if(is_endpoint && gpdb.getPins()[pin_id].getType() != 'c' && gpdb.getPins()[pin_id].getDirection() != 'o'){
                    endpoints_id.push_back(pin_id);
                    // std::cout<< gpdb.getPins()[pin_id].getName()<<std::endl;
                }
                pin_id2timing_arc_list_end[2 * pin_id + el] = timing_arc_id_map.size();
            }
        }
    }
    num_POs = endpoints_id.size();
    
    //  Construct Connectivity Graph
    //
    for (int i = 0; i < num_pins; i++) total_num_fanouts += STA_pins[i]->fanout_pin_ids.size();

    pin_fanout_list_end.resize(num_pins + 1);
    pin_fanout_list_end[0] = 0;
    pin_num_fanin.resize(num_pins);
    pin_fanout_list.resize(total_num_fanouts);

    index_type ptr = 0;
    index_type last_idx = 0;
    for (index_type i = 0; i < static_cast<index_type>(num_pins); i++) {
        for (auto fanout_pin_id : STA_pins[i]->fanout_pin_ids) pin_fanout_list[ptr++] = fanout_pin_id;
        last_idx += STA_pins[i]->fanout_pin_ids.size();
        pin_fanout_list_end[i + 1] = last_idx;
        pin_num_fanin[i] = STA_pins[i]->fanin_pin_ids.size();
    }
    for (int i = 0; i < num_pins; i++) {
        if (pin_num_fanin[i] == 0) pin_frontiers.push_back(i);
    }

    pin_forward_arc_list_end.push_back(0);
    pin_backward_arc_list_end.push_back(0);
    for (index_type i = 0; i < static_cast<index_type>(num_pins); i++) {
        for (auto fanout_arc : STA_pins[i]->timing_arc_out) {
            pin_forward_arc_list.push_back(fanout_arc);
        }
        pin_forward_arc_list_end.push_back(pin_forward_arc_list.size());
        for (auto fanin_arc : STA_pins[i]->timing_arc_in) {
            pin_backward_arc_list.push_back(fanin_arc);
        }
        pin_backward_arc_list_end.push_back(pin_backward_arc_list.size());
    }

    // gputimer arrays
    auto device = timing_raw_db.node_size.device();
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    // Gate Sizing Initialization
    // Timer graph topology variables
    timing_raw_db.pin_forward_arc_list = torch::from_blob(pin_forward_arc_list.data(), {static_cast<index_type>(pin_forward_arc_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_forward_arc_list_end = torch::from_blob(pin_forward_arc_list_end.data(), {static_cast<index_type>(pin_forward_arc_list_end.size())}, options).contiguous().to(device);
    timing_raw_db.pin_backward_arc_list = torch::from_blob(pin_backward_arc_list.data(), {static_cast<index_type>(pin_backward_arc_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_backward_arc_list_end = torch::from_blob(pin_backward_arc_list_end.data(), {static_cast<index_type>(pin_backward_arc_list_end.size())}, options).contiguous().to(device);
    timing_raw_db.timing_arc_from_pin_id = torch::from_blob(timing_arc_from_pin_id.data(), {static_cast<index_type>(timing_arc_from_pin_id.size())}, options).contiguous().to(device);
    timing_raw_db.timing_arc_to_pin_id = torch::from_blob(timing_arc_to_pin_id.data(), {static_cast<index_type>(timing_arc_to_pin_id.size())}, options).contiguous().to(device);
    timing_raw_db.pin_num_fanin = torch::from_blob(pin_num_fanin.data(), {static_cast<index_type>(pin_num_fanin.size())}, options).contiguous().to(device);
    timing_raw_db.pin_fanout_list = torch::from_blob(pin_fanout_list.data(), {static_cast<index_type>(pin_fanout_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_fanout_list_end = torch::from_blob(pin_fanout_list_end.data(), {static_cast<index_type>(pin_fanout_list_end.size())}, options).contiguous().to(device);

    // Timer timing liberty variables
    timing_raw_db.arc_types = torch::from_blob(arc_types.data(), {static_cast<int>(arc_types.size())}, options).contiguous().to(device);
    timing_raw_db.timing_arc_id_map = torch::from_blob(timing_arc_id_map.data(), {static_cast<int>(timing_arc_id_map.size())}, options).contiguous().to(device);
    timing_raw_db.arc_id2test_id = torch::from_blob(arc_id2test_id.data(), {static_cast<int>(arc_id2test_id.size())}, options).contiguous().to(device);
    timing_raw_db.test_id2_arc_id = torch::from_blob(test_id2_arc_id.data(), {static_cast<int>(test_id2_arc_id.size())}, options).contiguous().to(device);
    timing_raw_db.endpoints_id = torch::from_blob(endpoints_id.data(), {static_cast<index_type>(endpoints_id.size())}, options).contiguous().to(device);

    timing_raw_db.pinSlew = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinImpulse = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.clock_net_pin = torch::zeros({num_pins}, torch::dtype(torch::kBool).device(torch::Device(device))).contiguous();
    torch::fill_(timing_raw_db.pinSlew, nanf(""));
    torch::fill_(timing_raw_db.pinRAT, nanf(""));
    torch::fill_(timing_raw_db.pinAT, nanf(""));
    torch::fill_(timing_raw_db.pinImpulse, nanf(""));
    torch::fill_(timing_raw_db.pinRootDelay, nanf(""));

    timing_raw_db.arcDelay = torch::zeros({num_arcs, 2 * NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinImpulse_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad_ratio = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay_ratio = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay_compensation = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    logger.info("Design info: %d pins, %d arcs, %d tests", num_pins, num_arcs, num_tests);
}
std::string escapeBrackets(const std::string& input) {
    size_t last_rbracket = input.find_last_of(']');
    if (last_rbracket == std::string::npos) {
        return input; 
    }
    std::string res = input;
    for (size_t pos = last_rbracket; pos != std::string::npos; pos = res.find_last_of("]", pos - 1)) {
        if(!std::isdigit(input[pos - 1])){ 
            res.insert(pos, "\\");
            pos = res.find_last_of("[", pos - 1);
            res.insert(pos, "\\");
        }
    }
    
    return res;
}

void GTDatabase::readSdc(sdc::SDC& sdc) {
    for (auto& command : sdc.commands) {
        std::visit(Functors{[this](auto&& cmd) { _read_sdc(cmd); }}, command);
    }

    // string clock_name = clocks.begin()->second.source_name();
    assert(clocks.size() != 0);
    string clock_name = gpdb.getPins()[clocks.begin()->second.source_id()].getName();
    float period = clocks.begin()->second.period();
    logger.info("clock: %s, period: %.2f", clock_name.c_str(), period);

    net_is_clock.resize(gpdb.getNets().size(), 0);
    // for (auto& gpnet : gpdb.getNets()) { // dqk: why we need to rule out CLK -> QN if CLK is not directly connected to clknet?
    //     if (gpnet.getName() == clock_name) {
    //         // net_is_clock[gpnet.getId()] = 1;
            
    //         for (auto& gppin : gpnet.pins()) {
    //             timing_raw_db.pinAT[gppin][0] = 0.0f;
    //             timing_raw_db.pinAT[gppin][1] = 0.0f;
    //             timing_raw_db.pinAT[gppin][2] = 0.0f;
    //             timing_raw_db.pinAT[gppin][3] = 0.0f;
    //             timing_raw_db.clock_net_pin[gppin] = true;
    //         }
    //         for(auto& arc_id : test_id2_arc_id){
    //             int from_pin = timing_arc_from_pin_id[arc_id];
    //             int to_pin = timing_arc_to_pin_id[arc_id];
    //             if(!timing_raw_db.clock_net_pin[from_pin].item<bool>() && !timing_raw_db.clock_net_pin[to_pin].item<bool>()){
    //                 timing_raw_db.arc_id2test_id[arc_id] = -1;
    //                 int clk_port_pin = gpdb.getPins()[from_pin].getType() == 'c' ? from_pin : to_pin;
    //                 for(int i = pin_forward_arc_list_end[clk_port_pin]; i < pin_forward_arc_list_end[clk_port_pin+1]; i++){
    //                     int timing_arc_id = pin_forward_arc_list[i];
    //                     if(arc_types[timing_arc_id] != 1)continue;
    //                     timing_raw_db.timing_arc_id_map[2 * timing_arc_id] = timing_raw_db.timing_arc_id_map[2 * timing_arc_id + 1] = -1;
    //                     // printf("add test on arc %d from pin %s to pin %s\n", arc_id2, pin_names[from_pin2].c_str(), pin_names[to_pin2].c_str());
    //                 }
    //                 // printf("remove test on arc %d from pin %s to pin %s\n", arc_id, pin_names[from_pin].c_str(), pin_names[to_pin].c_str());
    //             }
    //         }
            
    //     }
    // }
    

    // set nan slew of PIs to half period
    for (auto& pi : primary_inputs) {
        if (torch::isnan(timing_raw_db.pinSlew[pi][0]).item<bool>()) timing_raw_db.pinSlew[pi][0] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][1]).item<bool>()) timing_raw_db.pinSlew[pi][1] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][2]).item<bool>()) timing_raw_db.pinSlew[pi][2] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][3]).item<bool>()) timing_raw_db.pinSlew[pi][3] = 0.0f;
        if (pin_names[pi] == clock_name) {
            timing_raw_db.pinSlew[pi][0] = 0.0f;
            timing_raw_db.pinSlew[pi][1] = 0.0f;
            timing_raw_db.pinSlew[pi][2] = 0.0f;
            timing_raw_db.pinSlew[pi][3] = 0.0f;
        }
        // if (torch::isnan(pinAT[pi][0]).item<bool>()) pinAT[pi][0] = 0.0f;
        // if (torch::isnan(pinAT[pi][1]).item<bool>()) pinAT[pi][1] = period / 2.0;
        // if (torch::isnan(pinAT[pi][2]).item<bool>()) pinAT[pi][2] = 0.0f;
        // if (torch::isnan(pinAT[pi][3]).item<bool>()) pinAT[pi][3] = period / 2.0;
    }

    assert(clocks.size() > 0);
    if (clocks.begin()->second.source_id() != -1) {
        int clock_pin_id = clocks.begin()->second.source_id();
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][0]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][0] = 0.0f;
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][1]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][1] = 0.0f;
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][2]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][2] = 0.0f;
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][3]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][3] = 0.0f;
        // if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][0]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][0] = 0.0f;
        // if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][1]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][1] = period / 2.0;
        // if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][2]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][2] = 0.0f;
        // if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][3]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][3] = period / 2.0;
    }
}

// Sets input delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetUnits& obj) {
    if (obj.time.has_value()) {
        auto s = *obj.time;
        if (s == "ps") sdc_time_unit = 1e-12;
        if (s == "ns") sdc_time_unit = 1e-9;
        if (s == "us") sdc_time_unit = 1e-6;
        if (s == "ms") sdc_time_unit = 1e-3;
        if (s == "s") sdc_time_unit = 1.0;
    }
    if (obj.capacitance.has_value()) {
        auto s = *obj.capacitance;
        if (s == "fF") sdc_cap_unit = 1e-15;
        if (s == "pF") sdc_cap_unit = 1e-12;
        if (s == "nF") sdc_cap_unit = 1e-9;
        if (s == "uF") sdc_cap_unit = 1e-6;
        if (s == "F") sdc_cap_unit = 1.0;
    }
    if (obj.resistance.has_value()) {
        auto s = *obj.resistance;
        if (s == "Ohm") sdc_res_unit = 1.0;
        if (s == "kOhm") sdc_res_unit = 1e3;
        if (s == "MOhm") sdc_res_unit = 1e6;
    }
    if (sdc_time_unit.has_value()) printf("sdc time unit: %.2E\n", *sdc_time_unit);
    if (sdc_cap_unit.has_value()) printf("sdc capacitance unit: %.2E\n", *sdc_cap_unit);
    if (sdc_res_unit.has_value()) printf("sdc resistance unit: %.2E\n", *sdc_res_unit);
}

// Sets input delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float delay = *obj.delay_value;
                                    if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinAT[pi][(el << 1) + rf] = delay;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(escapeBrackets(port)); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float delay = *obj.delay_value;
                                        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinAT[itr->second][(el << 1) + rf] = delay;
                                    }
                                } else {
                                    std::cout << obj.command << " : port " << std::quoted(port) << " not found" << std::endl;
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

void GTDatabase::_read_sdc(sdc::SetPropagatedClock& obj){
    assert(obj.object_list);
}
void GTDatabase::_read_sdc(sdc::SetClockLatency& obj){
    assert(obj.delay && obj.object_list);
    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);
    std::visit(Functors{[&](sdc::GetPins& get_pins){
                            for (auto& pin : get_pins.pins) {
                                if (auto itr = std::find(pin_names.begin(), pin_names.end(), pin.replace(pin.find_last_of("/"), 1, ":")); itr != pin_names.end()) {
                                    int pin_id = itr - pin_names.begin();
                                    assert(pin_id < (int)pin_names.size());
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float delay = *obj.delay;
                                        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinAT[pin_id][(el << 1) + rf] = delay;
                                    }
                                } else {
                                    std::cout << obj.command << " : pin " << std::quoted(pin) << " not found" << std::endl;
                                }
                            }
                        },
                        [](auto&&) { assert(false);  }},
               *obj.object_list);   
}
// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputTransition& obj) {
    assert(obj.transition && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = *obj.transition;
                                    if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinSlew[pi][(el << 1) + rf] = transition;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(escapeBrackets(port)); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = *obj.transition;
                                        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinSlew[itr->second][(el << 1) + rf] = transition;
                                    }
                                } else {
                                    std::cout << obj.command << " : port " << quoted(port) << " not found" << std::endl;
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_list);
}

// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetDrivingCell& obj) {
    assert((obj.transitions[0] || obj.transitions[1]) && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = *obj.transitions[el];
                                    if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinSlew[pi][(el << 1) + rf] = transition;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(escapeBrackets(port)); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = *obj.transitions[el];
                                        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinSlew[itr->second][(el << 1) + rf] = transition;
                                    }
                                } else {
                                    std::cout << obj.command << " : port " << std::quoted(port) << " not found" << std::endl;
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_list);
}

// Sets output delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetOutputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    if (clocks.find(obj.clock) == clocks.end()) {
        std::cout << obj.command << " : clock " << std::quoted(obj.clock) << " not found" << std::endl;
        return;
    }

    auto& clock = clocks.at(obj.clock);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : primary_outputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float delay = *obj.delay_value;
                                    if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinRAT[po][(el << 1) + rf] = el == MIN ? -delay : clock.period() - delay;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_output2pin_id.find(escapeBrackets(port)); itr != primary_output2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float delay = *obj.delay_value;
                                        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinRAT[itr->second][(el << 1) + rf] = el == MIN ? -delay : clock.period() - delay;
                                    }
                                } else {
                                    std::cout << obj.command << " : port " << std::quoted(port) << " not found" << std::endl;
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

// Sets the load attribute to a specified value on specified ports and nets.
void GTDatabase::_read_sdc(sdc::SetLoad& obj) {
    assert(obj.value && obj.objects);

    auto mask = sdc::TimingMask(obj.min, obj.max, std::nullopt, std::nullopt);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : primary_outputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float load = *obj.value;
                                    if (sdc_res_unit.has_value()) load = load * *sdc_res_unit / res_unit;
                                    timing_raw_db.pinLoad[po][(el << 1) + rf] = load;
                                    pin_capacitance[6 * po + el * 2 + rf] = load;
                                    pin_capacitance[6 * po + 4 + el] = load;
                                }                         
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_output2pin_id.find(escapeBrackets(port)); itr != primary_output2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float load = *obj.value;
                                        if (sdc_res_unit.has_value()) load = load * *sdc_res_unit / res_unit;
                                        timing_raw_db.pinLoad[itr->second][(el << 1) + rf] = load;
                                        pin_capacitance[6 * itr->second + el * 2 + rf] = load;
                                        pin_capacitance[6 * itr->second + 4 + el] = load;
                                    }
                                } else {
                                    std::cout << obj.command << " : port " << std::quoted(port) << " not found" << std::endl;
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.objects);
}

void GTDatabase::_read_sdc(sdc::CreateClock& obj) {
    assert(obj.period && !obj.name.empty());

    // create clock from given sources
    if (obj.port_pin_list) {
        std::visit(Functors{[&](sdc::GetPorts& get_ports) {
                                auto& ports = get_ports.ports;
                                assert(ports.size() == 1);
                                if (auto itr = primary_input2pin_id.find(escapeBrackets(ports.front())); itr != primary_input2pin_id.end()) {
                                    clocks.try_emplace(obj.name, obj.name, itr->second, *obj.period);
                                } else {
                                    std::cout << obj.command << " : port " << std::quoted(ports.front()) << " not found" << std::endl;
                                }
                            },
                            [](auto&&) { assert(false); }},
                   *obj.port_pin_list);
    }
    // create virtual clock
    else {
        clocks.try_emplace(obj.name, obj.name, *obj.period);
    }
}

TimingTorchRawDB::TimingTorchRawDB(torch::Tensor node_lpos_init_,
                                   torch::Tensor node_size_,
                                   torch::Tensor pin_rel_lpos_,
                                   torch::Tensor pin_id2node_id_,
                                   torch::Tensor pin_id2net_id_,
                                   torch::Tensor node2pin_list_,
                                   torch::Tensor node2pin_list_end_,
                                   torch::Tensor hyperedge_list_,
                                   torch::Tensor hyperedge_list_end_,
                                   torch::Tensor net_mask_,
                                   int num_movable_nodes_,
                                   float scale_factor_,
                                   int microns_,
                                   float wire_resistance_per_micron_,
                                   float wire_capacitance_per_micron_) {
    node_lpos_init = node_lpos_init_;
    node_size = node_size_;
    pin_rel_lpos = pin_rel_lpos_;

    node_size_x = node_size.index({"...", 0}).clone().contiguous();
    node_size_y = node_size.index({"...", 1}).clone().contiguous();
    init_x = node_lpos_init.index({"...", 0}).clone().contiguous();
    init_y = node_lpos_init.index({"...", 1}).clone().contiguous();
    pin_offset_x = pin_rel_lpos.index({"...", 0}).clone().contiguous();
    pin_offset_y = pin_rel_lpos.index({"...", 1}).clone().contiguous();
    x = init_x.clone().contiguous();
    y = init_y.clone().contiguous();

    num_nodes = node_size.size(0);
    num_pins = pin_id2node_id_.size(0);
    num_nets = hyperedge_list_end_.size(0);
    num_movable_nodes = num_movable_nodes_;
    net_mask = net_mask_;

    pinAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(node_size.device()))).contiguous();
    pinRAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(node_size.device()))).contiguous();
    at_prefix_pin = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))).contiguous();
    at_prefix_arc = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))).contiguous();
    at_prefix_attr = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))).contiguous();

    flat_node2pin_start_map = torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))), node2pin_list_end_}, 0).to(torch::kInt32).contiguous();
    flat_node2pin_map = node2pin_list_.to(torch::kInt32);
    pin2node_map = pin_id2node_id_.to(torch::kInt32);

    flat_net2pin_start_map = torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::Device(node_size.device()))), hyperedge_list_end_}, 0).to(torch::kInt32).contiguous();
    flat_net2pin_map = hyperedge_list_.to(torch::kInt32);
    pin2net_map = pin_id2net_id_.to(torch::kInt32);

    num_threads = std::max(6, 1);
    scale_factor = scale_factor_;
    microns = microns_;
    wire_resistance_per_micron = wire_resistance_per_micron_;
    wire_capacitance_per_micron = wire_capacitance_per_micron_;
}

void TimingTorchRawDB::commit_from(torch::Tensor x_, torch::Tensor y_) {
    // commit external pos to original pos
    init_x.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    init_y.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    x.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    y.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
}

torch::Tensor TimingTorchRawDB::get_curr_cposx() { return x + node_size_x / 2; }
torch::Tensor TimingTorchRawDB::get_curr_cposy() { return y + node_size_y / 2; }
torch::Tensor TimingTorchRawDB::get_curr_lposx() { return x; }
torch::Tensor TimingTorchRawDB::get_curr_lposy() { return y; }

}  // namespace gt