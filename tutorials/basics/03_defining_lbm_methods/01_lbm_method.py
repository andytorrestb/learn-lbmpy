from lbmpy.session import *

if __name__ == "__main__":
    from lbmpy.creationfunctions import create_lb_method
    lbm_config = LBMConfig(stencil=Stencil.D2Q9, method=Method.MRT_RAW, zero_centered=False)

    method = create_lb_method(lbm_config=lbm_config)
    # check also method='srt', 'trt', 'mrt'
    print(method)

    print(method.moment_matrix)

    method.stencil.plot()
    plt.savefig('stencil_plot.png')

    print(method.stencil)

    rr = [sp.Symbol('omega_shear'), sp.Symbol('omega_bulk'), sp.Symbol('omega_3'), sp.Symbol('omega_4')]

    lbm_config = LBMConfig(stencil=Stencil.D2Q9, method=Method.MRT, weighted=True,
                        relaxation_rates=rr, zero_centered=False)
    weighted_ortho_mrt = create_lb_method(lbm_config=lbm_config)
    print(weighted_ortho_mrt)

    lbm_config = LBMConfig(stencil=Stencil.D2Q9, method=Method.MRT, weighted=False,
                        relaxation_rates=rr, zero_centered=False)
    ortho_mrt = create_lb_method(lbm_config=lbm_config)
    print(ortho_mrt)

    print(ortho_mrt.is_orthogonal)
    print(weighted_ortho_mrt.is_weighted_orthogonal)

    lbm_config = LBMConfig(stencil=Stencil.D2Q9, method=Method.CENTRAL_MOMENT, equilibrium_order=4,
                       compressible=True, relaxation_rates=rr)

    central_moment_method = create_lb_method(lbm_config)
    print(central_moment_method)

    print(central_moment_method.shift_matrix)

    from lbmpy.methods import mrt_orthogonal_modes_literature
    from lbmpy.moments import MOMENT_SYMBOLS

    x, y, z = MOMENT_SYMBOLS

    moments = mrt_orthogonal_modes_literature(LBStencil(Stencil.D2Q9), is_weighted=True)
    print(moments)

    method = create_lb_method(LBMConfig(stencil=Stencil.D2Q9, method=Method.MRT, nested_moments=moments,
                            relaxation_rates=rr, continuous_equilibrium=False, zero_centered=False))
    print(method)

    ch = create_channel(domain_size=(100, 30), lb_method=method, u_max=0.05,
                    kernel_params={'omega_bulk': 1.8, 'omega_shear': 1.4, 'omega_3': 1.0, 'omega_4': 1.0})
    ch.run(500)
    plt.figure(dpi=200)
    plt.vector_field(ch.velocity[:, :])
    plt.savefig('velocity_field_channel.png')