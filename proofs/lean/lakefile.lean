import Lake
open Lake DSL

package «stepback-soundness» where
  name := "stepback-soundness"

lean_lib «Stepback» where
  roots := #[`Stepback.Soundness]

@[default_target]
lean_exe «check» where
  root := `Main
