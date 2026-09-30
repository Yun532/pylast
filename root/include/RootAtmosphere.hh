#pragma once
#include "AtmosphereModel.hh"
#include "ROOT/RVec.hxx"
#include "TFile.h"
#include "TTree.h"
#include <optional>
#include <stdexcept>

// Shared by LACTsim inputs and pylast reconstruction outputs; absence is valid
// for old files, but a present, malformed table must never become a fallback.
inline std::optional<TableAtmosphereModel> read_root_atmosphere(TFile& file)
{
    auto* object = file.Get("cfg/atmosphere_model");
    if (!object) return std::nullopt;
    auto* tree = dynamic_cast<TTree*>(object);
    if (!tree || tree->GetEntries() != 1)
        throw std::runtime_error("cfg/atmosphere_model must be a one-entry TTree");
    ROOT::VecOps::RVec<double> alt, density, depth, index;
    auto *a = &alt, *r = &density, *t = &depth, *n = &index;
    const char* names[] = {"alt_km", "rho", "thick", "refidx_m1"};
    ROOT::VecOps::RVec<double>** pointers[] = {&a, &r, &t, &n};
    for (int i = 0; i < 4; ++i) {
        if (!tree->GetBranch(names[i]) || tree->SetBranchAddress(names[i], pointers[i]) < 0) {
            tree->ResetBranchAddresses();
            throw std::runtime_error(std::string("Invalid atmosphere branch: ") + names[i]);
        }
    }
    const auto bytes = tree->GetEntry(0);
    tree->ResetBranchAddresses();
    if (bytes <= 0 || a->size() != r->size() || a->size() != t->size() || a->size() != n->size())
        throw std::runtime_error("Invalid atmosphere arrays in ROOT file");
    TableAtmosphereModel profile(a->size(), a->data(), r->data(), t->data(), n->data());
    profile.input_filename = std::string(file.GetName()) + "#cfg/atmosphere_model";
    return profile;
}
