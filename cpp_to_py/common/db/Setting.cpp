#include "Setting.h"

namespace db {
void Setting::reset() {
    Format = "";
    BookshelfVariety = "";
    BookshelfAux = "";
    BookshelfPl = "";
    DefFile = "";
    LefFile = "";
    LefCell = "";
    LefTech = "";
    LefFiles.clear();
    LibFiles.clear();
    Constraints = "";

    CellLib = "";
    CellLib_MIN = "";
    CellLib_MAX = "";

    Verilog = "";
    OutputFile = "";
    
    EnablePG = true;
    EnableIOPin = true;
    EnableFence = true;
    liteMode = true;
    random_place = false;
}

Setting setting;

}  //   namespace db
