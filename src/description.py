import seml.typer as typer
import os

app = typer.Typer(
    no_args_is_help=True,
    help='Manage descriptions of the experiments in a collection.',
    chain=bool(os.environ.get('_SEML_COMPLETE'))
)

@app.command("set")
@typer.restrict_collection()
def reload_sources_command(
    ctx: typer.Context,
    sacred_id: typer.SacredIdAnnotation = None,
    filter_states: typer.FilterStatesAnnotation = None,
    filter_dict: typer.FilterDictAnnotation = None,
    batch_id: typer.BatchIdAnnotation = None,
    yes: typer.YesAnnotation = False,
):
    """
    Reload stashed source files.
    """
    print(f'Set description', ctx.obj['collection'], sacred_id, filter_states, filter_dict, batch_id, yes)
