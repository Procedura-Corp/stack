// AlgorithmicMemory.hpp – C++20 header‑only rewrite (head‑window traversal added)
// ---------------------------------------------------------------------------
// © 2025 Janusz Jeremiasz Filipiak – MIT Licence
// ---------------------------------------------------------------------------
//  Major changes in this revision:
//   • Implements a **3×3×3 head window** identical to the original Python
//     AlgorithmicMemory head‑buffer so you can pan the lattice cheaply.
//   • Adds `move_head_abs()`, `cursor()`, and internal `initialize_head()`
//     which create missing scaffolds on demand and wire x/y/z neighbours.
//   • Keeps public API stable – existing `insert()` calls still work; now you
//     can call `move_head_abs(x,y,z)` and then `cursor()` to read/write the
//     central DBit at (x,y,z).
//   • Uses `std::array` for the fixed‑size head cube and constexpr params.
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace am {
// ─────────────────────────────────────────────────────────────────────────────
//  Forward declarations
// ─────────────────────────────────────────────────────────────────────────────
struct LBit;
struct DBit;

// ─────────────────────────────────────────────────────────────────────────────
//  TimeRoot – monotonic ID generator (thread‑safe)
// ─────────────────────────────────────────────────────────────────────────────
struct TimeRoot {
    using Id = std::uint64_t;

    TimeRoot() noexcept : id_{ ++counter_ } {}
    explicit TimeRoot(Id raw) noexcept : id_{ raw } {}

    [[nodiscard]] Id value() const noexcept { return id_; }
    std::string str() const { return "tr_" + std::to_string(id_); }
private:
    Id id_;
    static std::atomic<Id> counter_;
};
inline std::atomic<TimeRoot::Id> TimeRoot::counter_{ 0 };

// ─────────────────────────────────────────────────────────────────────────────
//  LBit – six‑dimensional node
// ─────────────────────────────────────────────────────────────────────────────
struct LBit : public std::enable_shared_from_this<LBit> {
    using Ptr = std::shared_ptr<LBit>;

    TimeRoot   tr;
    std::string data;   // payload (binary string or tag)
    Ptr other;          // ring link (dimension 0)
    Ptr x, y, z;        // spatial links (dims 1‑3)

    explicit LBit(std::string payload) : tr{}, data{ std::move(payload) } {}

    std::string repr() const {
        std::ostringstream oss;
        oss << tr.str() << "_LBit(data=" << data << ")";
        return oss.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  DBit – pair of opposite LBits anchored at (x,y,z)
// ─────────────────────────────────────────────────────────────────────────────
struct DBit {
    using Ptr   = std::shared_ptr<DBit>;
    using Coord = std::tuple<int,int,int>;

    LBit::Ptr lbit0;
    LBit::Ptr lbit1;
    int       x{0}, y{0}, z{0};

    DBit(LBit::Ptr a, LBit::Ptr b, int xi, int yi, int zi)
        : lbit0{ std::move(a) }, lbit1{ std::move(b) }, x{ xi }, y{ yi }, z{ zi } {
        lbit0->other = lbit1;
        lbit1->other = lbit0;
    }

    void set_3d_neighbourship() noexcept {
        // No‑op.  Caller wires neighbours when building the head window.
    }

    [[nodiscard]] Coord coord() const noexcept { return {x,y,z}; }
    std::string repr() const {
        std::ostringstream oss;
        oss << "DBit(" << x << "," << y << "," << z << ")\n  "
            << lbit0->repr() << "\n  " << lbit1->repr();
        return oss.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  LatticeStack – vertical ring (one coordinate)
// ─────────────────────────────────────────────────────────────────────────────
class LatticeStack {
public:
    explicit LatticeStack(const std::shared_ptr<DBit>& scaffold) : head_{ scaffold->lbit0 } {}

    std::shared_ptr<DBit> push(std::string payload) {
        auto new0 = std::make_shared<LBit>(std::move(payload));
        auto new1 = std::make_shared<LBit>("time_root");
        auto db   = std::make_shared<DBit>(new0,new1,0,0,0); // coord fixed later
        // splice into ring
        auto wall = head_->other;
        new0->other = head_;
        head_->other = new0;
        new1->other = wall;
        wall->other = new1;
        head_ = new0;
        return db;
    }

    struct iterator {
        using Ptr = LBit::Ptr;
        Ptr cur{}, start{}; bool first{true};
        iterator() = default; explicit iterator(Ptr n):cur{n},start{n}{}
        Ptr operator*() const { return cur; }
        iterator& operator++(){ if(cur){ cur=cur->other; if(cur==start)cur=nullptr; } return *this; }
        bool operator==(const iterator& o) const { return cur==o.cur; }
        bool operator!=(const iterator& o) const { return !(*this==o); }
    };
    iterator begin() const { return iterator{ head_ }; }
    iterator end()   const { return iterator{}; }
private:
    LBit::Ptr head_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  AlgorithmicMemory – full lattice orchestrator with head window
// ─────────────────────────────────────────────────────────────────────────────
class AlgorithmicMemory {
public:
    using Coord = std::tuple<int,int,int>;
    static constexpr int HEAD_DIM = 3;
    static constexpr int MAX_COMP_DIM = 50000; // safety bound
    using HeadArray = std::array<std::array<std::array<std::shared_ptr<DBit>,HEAD_DIM>,HEAD_DIM>,HEAD_DIM>;

    AlgorithmicMemory(){
        initialize_head(-1,-1,-1);      // bootstrap window around (0,0,0)
        move_head_abs(0,0,0);
    }

    // ───────── public lattice I/O ─────────
    std::shared_ptr<DBit> insert(std::string data,int x,int y,int z){
        auto sc = scaffold_at(x,y,z);
        LatticeStack stack{ sc };
        auto nb = stack.push(std::move(data));
        nb->x=x; nb->y=y; nb->z=z;
        dbit_list_.push_back(nb);
        relink_coord(x,y,z);
        return nb;
    }

    std::shared_ptr<DBit> move_head_abs(int x,int y,int z){
        const Coord desired_origin{ x-1, y-1, z-1 };
        if(desired_origin == head_origin_) return cursor();
        auto [ox,oy,oz] = head_origin_;
        auto [dx,dy,dz] = desired_origin;
        if(std::abs(dx-ox)+std::abs(dy-oy)+std::abs(dz-oz) > MAX_COMP_DIM)
            throw std::out_of_range("AlgorithmicMemory: head move too large");
        initialize_head(std::get<0>(desired_origin), std::get<1>(desired_origin), std::get<2>(desired_origin));
        head_origin_ = desired_origin;
        return cursor();
    }

    std::shared_ptr<DBit> cursor() const noexcept { return head_[1][1][1]; }
    std::shared_ptr<DBit> scaffold_at(int x,int y,int z){
        Coord key{x,y,z};
        auto [it,created] = dbits_.try_emplace(key,nullptr);
        if(created){
            auto l0 = std::make_shared<LBit>("3d"+coord_str(key));
            auto l1 = std::make_shared<LBit>("time_root");
            it->second = std::make_shared<DBit>(l0,l1,x,y,z);
        }
        return it->second;
    }

private:
    // Build / refresh 3×3×3 window whose origin is (xs,ys,zs)
    void initialize_head(int xs,int ys,int zs){
        for(int i=0;i<HEAD_DIM;++i){
            for(int j=0;j<HEAD_DIM;++j){
                for(int k=0;k<HEAD_DIM;++k){
                    int gx=xs+i, gy=ys+j, gz=zs+k;
                    auto db = scaffold_at(gx,gy,gz);
                    head_[i][j][k]=db;
                    // wire negative direction neighbours (lbit0 side)
                    if(i>0){  // x‑link
                        auto left = head_[i-1][j][k];
                        left->lbit1->x = db->lbit0;
                        db->lbit0->x   = left->lbit1;
                    }
                    if(j>0){  // y‑link
                        auto back = head_[i][j-1][k];
                        back->lbit1->y = db->lbit0;
                        db->lbit0->y   = back->lbit1;
                    }
                    if(k>0){  // z‑link
                        auto down = head_[i][j][k-1];
                        down->lbit1->z = db->lbit0;
                        db->lbit0->z   = down->lbit1;
                    }
                }
            }
        }
    }

    // vertical ring relink (same as previous version)
    void relink_coord(int x,int y,int z){
        std::vector<std::shared_ptr<DBit>> stack;
        for(auto& db: dbit_list_) if(db->x==x && db->y==y && db->z==z) stack.push_back(db);
        if(stack.size()<2) return;
        std::sort(stack.begin(),stack.end(),[](auto&a,auto&b){return a->lbit0->tr.value()<b->lbit0->tr.value();});
        auto it = std::find_if(stack.begin(),stack.end(),[](auto&d){return d->lbit0->data.starts_with("3d");});
        if(it!=stack.begin()&&it!=stack.end()) std::rotate(stack.begin(),it,stack.end());
        const size_t n=stack.size();
        for(size_t i=0;i<n;++i){
            auto nxt=stack[(i+1)%n];
            stack[i]->lbit0->other=nxt->lbit0;
            stack[i]->lbit1->other=nxt->lbit1;
        }
    }

    static std::string coord_str(const Coord& c){ auto[x,y,z]=c; std::ostringstream o; o<<x<<"_"<<y<<"_"<<z; return o.str(); }

    // storage
    std::unordered_map<Coord,std::shared_ptr<DBit>,std::hash<Coord>> dbits_;
    std::vector<std::shared_ptr<DBit>> dbit_list_;

    // 3×3×3 window
    HeadArray head_{};
    Coord head_origin_{-1,-1,-1};
};

} // namespace am

/*
──────────────────────────────── HOW TO PAN THE WINDOW ─────────────────────────

#include "AlgorithmicMemory.hpp"

int main(){
    am::AlgorithmicMemory mem;
    mem.insert("centre",0,0,0);

    // Move the head so (2,3,4) ends up at centre of window
    auto centre = mem.move_head_abs(2,3,4);
    std::cout << "Current head @ (2,3,4): " << centre->repr() << "\n";

    // read the DBit directly underneath (2,2,4):
    auto below = mem.scaffold_at(2,2,4);
    std::cout << below->repr() << "\n";
}
*/
