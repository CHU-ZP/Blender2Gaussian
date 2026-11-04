"""Contracts for the selected surface initializer and its CLI profile."""
import pytest

from scripts.initialize_blend_surface import covariance, parse_args, prepare_importance


def test_default_entry_selects_retained_backend_and_spacing_profile():
    args = parse_args(['data/sk_00918.blend'])
    assert args.base_spacing == .024 and args.budget is None
    assert (args.texture_weight, args.curvature_weight, args.detail_weight) == (2., 1., .25)
    assert (args.overlap, args.opacity, args.seed) == (.8, .99, 3089248367)
    # Validated default: seam-aware boundary limits and hidden-layer removal.
    assert (args.boundary_sigmas, args.seam_sigmas) == (2., 1.)
    assert args.visibility and args.scene_type == 'auto'
    assert (args.visibility_max_gap, args.visibility_reach, args.visibility_grid) == (.007, 4., .25)
    assert covariance.__module__ == 'mesh_to_gs.surface'
    assert prepare_importance.__module__ == 'mesh_to_gs.adaptive'


def test_explicit_budget_disables_automatic_spacing():
    args = parse_args(['data/sk_00918.blend', '--budget', '20000'])
    assert args.budget == 20000 and args.base_spacing is None


def test_cli_spacing_and_material_parameters_override_profile():
    args = parse_args(['data/sk_00918.blend', '--base-spacing', '.03',
                       '--texture-weight', '1', '--overlap', '.7', '--opacity', '.98'])
    assert (args.base_spacing, args.texture_weight, args.overlap, args.opacity) == (.03, 1., .7, .98)


def test_unsupported_profile_keys_are_rejected(tmp_path, capsys):
    profile = tmp_path / "unsupported.json"
    profile.write_text('{"base_spacing": 0.024, "directional_support": true}')
    with pytest.raises(SystemExit):
        parse_args(['data/sk_00918.blend', '--calibration-profile', str(profile)])
    assert 'Unsupported profile keys' in capsys.readouterr().err


def test_previous_behaviour_remains_selectable():
    args = parse_args(['data/sk_00918.blend', '--no-visibility', '--seam-sigmas', '2'])
    assert not args.visibility and args.seam_sigmas == args.boundary_sigmas


def test_build_model_keeps_default_pipeline_and_argument_passthrough():
    from scripts.build_model import parse_args as build_args
    default = build_args(['scene.blend', '--output-dir', 'out'])
    assert default.init == default.refine == []
    assert default.sampling == 'sampled'
    chosen = build_args(['scene.blend', '--output-dir', 'out',
                         '--sampling', 'legacy',
                         '--init-args', '--budget 20000', '--refine-args', '--steps 100'])
    assert chosen.init == ['--budget', '20000'] and chosen.refine == ['--steps', '100']
    with pytest.raises(SystemExit):
        build_args(['scene.blend', '--output-dir', 'out', '--refine-method', 'render'])
