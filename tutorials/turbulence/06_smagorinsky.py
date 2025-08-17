from lbmpy.session import *
from lbmpy.relaxationrates import *

from pystencils.simp import sympy_cse

from lbmpy.chapman_enskog import ChapmanEnskogAnalysis, CeMoment
from lbmpy.chapman_enskog.chapman_enskog import remove_higher_order_u

def second_order_moment_tensor(function_values, stencil):
    assert len(function_values) == len(stencil)
    dim = len(stencil[0])
    return sp.Matrix(dim, dim, lambda i, j: sum(c[i] * c[j] * f for f, c in zip(function_values, stencil)))


def frobenius_norm(matrix, factor=1):
    return sp.sqrt(sum(i*i for i in matrix) * factor)

def smagorinsky_equations(ω_0, ω_total, method):
    f_neq = sp.Matrix(method.pre_collision_pdf_symbols) - method.get_equilibrium_terms()
    return [sp.Eq(τ_0, 1 / ω_0),
            sp.Eq(Π, frobenius_norm(second_order_moment_tensor(f_neq, method.stencil), factor=2)),
            sp.Eq(ω_total, 1 / τ_val)]

def get_Π_1(ce_analysis, component):
    val = ce_analysis.higher_order_moments[component]
    return remove_higher_order_u(val.expand())

if __name__ == "__main__":
    τ_0, ρ, ω, ω_total, ω_0 = sp.symbols("tau_0 rho omega omega_total omega_0", positive=True, real=True)
    ν_0, C_S, S, Π = sp.symbols("nu_0, C_S, |S|, Pi", positive=True, real=True)
    print(f"ω_0 = {ω_0}")
    Seq = sp.Eq(S, 3 * ω / 2 * Π)
    print(f"Seq = {Seq}")

    tau = relaxation_rate_from_lattice_viscosity(ν_0 + C_S ** 2 * S)
    print(f"tau = {tau}")

    Seq2 = Seq.subs(ω, relaxation_rate_from_lattice_viscosity(ν_0 + C_S **2 * S ))
    print(f"Seq2 = {Seq2}")

    solveRes = sp.solve(Seq2, S)
    assert len(solveRes) == 1
    SVal = solveRes[0]
    SVal = SVal.subs(ν_0, lattice_viscosity_from_relaxation_rate(1 / τ_0)).expand()
    print(f"SVal = {SVal}")

    τ_val = 1 / (relaxation_rate_from_lattice_viscosity(lattice_viscosity_from_relaxation_rate(1/τ_0) + C_S**2 * SVal)).cancel()
    print(f"τ_val = {τ_val}")

    smagEq = smagorinsky_equations(ω_0, ω_total, create_lb_method())
    print(f"smagEq = {smagEq}")

    lbm_config = LBMConfig(stencil=Stencil.D2Q9, method=Method.MRT, force=(1e-6, 0),
                       force_model=ForceModel.LUO, relaxation_rates=[ω, 1.9, 1.9, 1.9])

    method = create_lb_method(lbm_config=lbm_config)
    print(f"method = {method}")

    optimization = {'simplification' : False}
    collision_rule = create_lb_collision_rule(lb_method=method, optimization=optimization)
    collision_rule = collision_rule.new_with_substitutions({ω: ω_total})

    collision_rule.subexpressions += smagorinsky_equations(ω, ω_total, method)
    collision_rule.topological_sort(sort_subexpressions=True, sort_main_assignments=False)
    print(collision_rule)

    ch = create_channel((300, 100), force=1e-6, collision_rule=collision_rule,
                    kernel_params={"C_S": 0.12, "omega": 1.999})
    ch.run(5000)

    plt.figure(dpi=200)
    plt.vector_field(ch.velocity[:, :])
    plt.savefig("velocity_field.png")
    print(f'max velocity = {np.max(ch.velocity[:, :])}')

    compressible_model = create_lb_method(stencil=Stencil.D2Q9, compressible=True, zero_centered=False)
    incompressible_model = create_lb_method(stencil=Stencil.D2Q9, compressible=False, zero_centered=False)

    ce_compressible = ChapmanEnskogAnalysis(compressible_model)
    ce_incompressible = ChapmanEnskogAnalysis(incompressible_model)

    Π_1_xy = CeMoment("\\Pi", moment_tuple=(1,1), superscript=1)
    Π_1_xx = CeMoment("\\Pi", moment_tuple=(2,0), superscript=1)
    Π_1_yy = CeMoment("\\Pi", moment_tuple=(0,2), superscript=1)
    components = (Π_1_xx, Π_1_yy, Π_1_xy)

    Π_1_xy_val = ce_compressible.higher_order_moments[Π_1_xy]
    print(Π_1_xy_val)

    print(remove_higher_order_u(Π_1_xy_val.expand()))

    print(tuple(get_Π_1(ce_compressible, Pi) for Pi in components))

    print(tuple(get_Π_1(ce_incompressible, Pi) for Pi in components))
