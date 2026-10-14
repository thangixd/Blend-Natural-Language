import argparse
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault('PYDANTIC_AI_NO_BANNER', '1')
os.environ.setdefault('TQDM_DISABLE', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.Agent.Actions import candidates, perform, plan_columns, plan_correlations
from src.Agent.Config import AgentConfig
from src.Agent.Dialogue import (Click, DialogueManager, LoadTable, Message, NoPreference, NotIt, Other, PlanCard,
                                Question, SetK, SetWording)
from src.Agent.Plans import KeywordFill, NaturalLanguageFill
from src.Agent.Questions import OTHER_KINDS
from src.Agent.Sampling import listening
from src.Agent.Text import Opaque

from typing import Any, List, Optional, Tuple, Union
from src.Agent.Dialogue import Response
from src.Agent.Sampling import Listener

PANE_HEIGHT = 600
PANE_CSS = """
<style>
.stMainBlockContainer {
    padding-top: 2.5rem;
}
/* Only a pixel height makes Streamlit scroll a container and follow new chat messages, and a
   "stretch" one grows with its content instead. The pixel height cannot follow the window, so
   it is overridden here, on the layout wrapper that holds it as its flex basis. The rem taken
   off is the page padding, the header and the chat input. Without the min-height a long chat
   keeps its pane at the fallback height on a window shorter than that. */
div[data-testid="stLayoutWrapper"]:has(> .st-key-chat_pane),
div[data-testid="stLayoutWrapper"]:has(> .st-key-query_panel) {
    flex-basis: calc(100dvh - 16.5rem);
    min-height: 0;
}
/* Streamlit wraps every container in a layout wrapper sized to its content, and a sticky
   element can only slide within its own parent - so the wrapper is what has to stick.
   The background must be a real colour: a transparent header lets the panel scroll
   through it, which is what made everything look overlapped. */
div[data-testid="stLayoutWrapper"]:has(> .st-key-app_header) {
    position: sticky;
    /* Clear of Streamlit's own 60px header bar, which sits at z-index 999990 and would
       otherwise swallow the clicks on these buttons. */
    top: 3.75rem;
    z-index: 100;
    background-color: $BACKGROUND$;
}
</style>
"""
SPINNER = {'load': 'reading the table', 'find': 'searching the lake', 'event': 'thinking',
           'act': 'combining the tables'}
STEPS = {'router': 'Reading your message', 'decomposer': 'Splitting the request into parts',
         'filler': 'Choosing the seekers and columns', 'single_call': 'Planning the search',
         'edit': 'Changing the plan',
         'verifier': 'Checking the plan', 'chat': 'Writing the reply', 'profiler': 'Profiling the columns'}
REDRAW = 0.3
THINKING_TAIL = 3000
PREVIEW_ROWS = 100
ACTED = {'use': 'Use', 'join': 'Join', 'union': 'Append the rows of', 'feature': 'Add a feature from',
         'fill': 'Fill blanks from'}
FAILED = 'That turn failed with an internal error, which the record keeps. Please try again, or rephrase your message.'


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.ini')
    parser.add_argument('--datalake', help='The directory of the indexed lake files, needed to find a query table '
                                           'and for the result actions.')
    return parser.parse_known_args()[0]


def _background() -> str:
    configured = st.get_option('theme.backgroundColor')
    if configured:
        return configured
    return '#0e1117' if st.context.theme.type == 'dark' else '#ffffff'


def _queue(kind: str, payload: Any) -> None:
    st.session_state['queued'] = (kind, payload)
    st.rerun()


def _current(position: int) -> bool:
    state = st.session_state
    return state['transcript'][position]['session'] == state['session']


def _content(text: Union[str, Opaque]) -> str:
    return text.for_prompt() if isinstance(text, Opaque) else str(text)


def _render(manager: Optional[DialogueManager], queued: Optional[Tuple[str, Any]], datalake: Optional[str]) -> None:
    busy = queued is not None
    answer = None
    for position, entry in enumerate(st.session_state['transcript']):
        if entry['kind'] == 'user':
            st.chat_message('user').write(_content(entry['text']))
            answer = None
            continue
        if answer is None:
            answer = st.chat_message('assistant')
        with answer:
            if entry['kind'] == 'note':
                st.caption(entry['text'])
            elif entry['kind'] == 'error':
                st.error(entry['text'])
            elif entry['kind'] == 'found':
                _candidates(entry['found'], position, busy)
            else:
                response: Response = entry['response']
                live = manager is not None and _current(position)
                if isinstance(response.shown, PlanCard):
                    _render_card(manager, response.shown, position, busy or not live, datalake)
                elif isinstance(response.shown, Question):
                    _render_question(manager, response.shown, position, busy or not live)
                elif response.shown is not None:
                    st.write(_content(response.shown.text))
                if response.chat:
                    st.write(_content(response.chat))
                for question in response.raised:
                    if question is not response.shown:
                        _render_question(manager, question, position, busy or not live)
    if busy and queued[0] == 'event':
        st.chat_message('user').write(_content(queued[1][0]))


def _render_card(manager: DialogueManager, card: PlanCard, position: int, off: bool, datalake: Optional[str]) -> None:
    st.markdown(f'**Plan v{card.version}:** {card.words}')
    if card.task:
        st.caption(f'Recognised as: {card.task}{" (assumed)" if "structure" in card.assumed else ""}')
    if card.assumed:
        st.caption(f'Assumed: {", ".join(card.assumed)}')
    if card.not_used:
        st.info(f'Not used: {", ".join(repr(_content(words)) for words in card.not_used)}')
    for change in card.changed:
        st.caption(f'Changed: {_content(change)}')
    for notice in card.notices:
        st.info(_content(notice))
    for line in card.discrepancies:
        st.warning(_content(line))
    with st.expander('Blend plan'):
        st.code('\n'.join(f'{node.name}: {node.operator}(k={node.k}) {node.description}'
                          + (f' <- {", ".join(node.inputs)}' if node.inputs else '') for node in card.nodes),
                language=None)
    if not card.tables:
        st.write('No tables found.')
    else:
        st.dataframe(pd.DataFrame([{'rank': rank, 'table id': table_id, 'file': name}
                                   for rank, (table_id, name) in enumerate(card.tables, 1)]), hide_index=True)

    off = off or card.version != (manager.plan.version if manager.plan else None)
    if off:
        return
    key = lambda name: f'{name}_{position}'
    for rank, (table_id, name) in enumerate(card.tables, 1):
        _found_table(manager, rank, name, key, datalake)
    with st.expander('Change the plan'):
        with st.form(key('k_form'), border=False):
            k = st.number_input('Number of tables', min_value=1, value=manager.plan.k.value, key=key('k'))
            if st.form_submit_button('Apply'):
                _queue('event', (f'Show {k} tables', SetK(int(k), version=card.version)))
        for part in manager.plan.parts():
            if isinstance(part.fill, (KeywordFill, NaturalLanguageFill)):
                words = isinstance(part.fill, KeywordFill)
                with st.form(key(f'{part.id}_wording'), border=False):
                    text = st.text_area(f'{part.id}: keyword words, one per line', '\n'.join(part.fill.words.value),
                                        key=key(f'{part.id}_words')) if words \
                        else st.text_input(f'{part.id}: description', part.fill.description.value,
                                           key=key(f'{part.id}_description'))
                    if st.form_submit_button('Apply'):
                        value = tuple(line.strip() for line in text.splitlines() if line.strip()) if words \
                            else text.strip()
                        _queue('event', (f'{part.id}: {", ".join(value) if words else value}',
                                         SetWording(part.id, value, version=card.version)))
    if card.not_it:
        with st.form(key('not_it'), border=False):
            text = st.text_input('What is wrong (optional)', key=key('not_it_text'))
            if st.form_submit_button('Not it'):
                _queue('event', (f'Not it{": " + text if text.strip() else ""}',
                                 NotIt(version=card.version, text=text.strip() or None)))


def _found_table(manager: DialogueManager, rank: int, name: str, key, datalake: Optional[str]) -> None:
    box = st.expander(f'{rank}. {name}', key=key(f'found_{rank}'), on_change='rerun')
    with box:
        if not datalake:
            st.caption('Start the app with --datalake <dir> to see and combine the found tables.')
            return
        table = manager.table
        keys = plan_columns(manager.plan)
        if st.button('Use as query table', key=key(f'use_{rank}')):
            _queue('act', ('use', rank, (), None))
        join, append = st.columns([3, 1], vertical_alignment='bottom')
        columns = join.multiselect('Join on your columns', table.columns, key=key(f'join_columns_{rank}'),
                                   default=list(keys[0]) if len(keys) == 1 else [])
        if join.button('Join', key=key(f'join_{rank}'), disabled=not columns):
            _queue('act', ('join', rank, tuple(columns), None))
        if append.button('Append rows', key=key(f'union_{rank}')):
            _queue('act', ('union', rank, (), None))
        correlations = plan_correlations(manager.plan)
        if len(correlations) == 1 and st.button(f'Add the feature that correlates most with {correlations[0][1]}, '
                                                f'joined on {correlations[0][0]}', key=key(f'feature_{rank}')):
            _queue('act', ('feature', rank, (), None))
        blanks = [column for column, profile in table.profile.items() if profile.empty]
        if blanks:
            column, by, go = st.columns([2, 2, 1], vertical_alignment='bottom')
            filled = column.selectbox('Fill the blanks of', blanks, key=key(f'fill_column_{rank}'))
            plan_keys = [found for found in (tuple(other for other in key if other != filled) for key in keys) if found]
            matched = by.multiselect('matched on', [other for other in table.columns if other != filled],
                                     key=key(f'fill_keys_{rank}_{filled}'),
                                     default=list(plan_keys[0]) if len(plan_keys) == 1 else [])
            if go.button('Fill blanks', key=key(f'fill_{rank}'), disabled=not matched):
                _queue('act', ('fill', rank, tuple(matched), filled))
        if box.open:
            st.caption(f'The first {PREVIEW_ROWS} rows.')
            st.dataframe(pd.read_csv(Path(datalake) / name, nrows=PREVIEW_ROWS, low_memory=False), hide_index=True)


def _render_question(manager: Optional[DialogueManager], question: Question, position: int, off: bool) -> None:
    st.info(_content(question.text))
    off = off or question.id not in {open_question.id for open_question in manager.questions.open}
    version = manager.plan.version if manager and manager.plan else None
    key = lambda name: f'{question.id}_{name}_{position}'
    labels = {option.id: _content(option.label) for option in question.options}
    for option in question.options:
        if st.button(labels[option.id], key=key(option.id), disabled=off):
            _queue('event', (labels[option.id], Click(question.id, option.id, version=version)))
    if question.kind == 'column' and len(question.options) > 1 and not off:
        together, go = st.columns([3, 1], vertical_alignment='bottom')
        picked = together.multiselect('Or several columns together', list(labels), format_func=labels.get,
                                      key=key('together'))
        if go.button('Pick together', key=key('pick_together'), disabled=len(picked) < 2):
            _queue('event', (', '.join(labels[option] for option in picked),
                             Click(question.id, tuple(picked), version=version)))
    if st.button('No preference', key=key('none'), disabled=off):
        _queue('event', ('No preference', NoPreference(question.id, version=version)))
    if question.kind in OTHER_KINDS and not off:
        with st.form(key('other'), border=False):
            text = st.text_input('Something else', key=key('other_text'))
            if st.form_submit_button('Answer') and text.strip():
                _queue('event', (text, Other(question.id, text, version=version)))


def _candidates(found: List[Tuple[int, Path]], position: int, busy: bool) -> None:
    if not found:
        st.write('The natural-language seeker found no table for that description.')
        return
    st.write(f'Found {len(found)} tables, best first. Open one to see its first rows.')
    for rank, (table_id, path) in enumerate(found, 1):
        box = st.expander(f'{rank}. {path.name}', key=f'candidate_{position}_{rank}', on_change='rerun')
        with box:
            st.caption(', '.join(map(str, pd.read_csv(path, nrows=0).columns)))
            if st.button('Use as query table', key=f'candidate_use_{position}_{rank}', disabled=busy):
                _queue('load', (path, True, path.name))
            if box.open:
                st.dataframe(pd.read_csv(path, nrows=PREVIEW_ROWS, low_memory=False), hide_index=True)


def _fill_panel(panel, busy: bool, datalake: Optional[str]) -> None:
    state = st.session_state
    manager: Optional[DialogueManager] = state['manager']
    with panel.container(height=PANE_HEIGHT, border=True, key='query_panel'):
        uploaded = st.file_uploader('Query table', type='csv', disabled=busy, key=f'upload_{state["round"]}')
        no_header = st.checkbox('File has no header row', key=f'no_header_{state["round"]}', disabled=busy)
        if uploaded is not None and uploaded.file_id != state.get('upload_id'):
            state['upload_id'] = uploaded.file_id
            path = Path(tempfile.mkdtemp()) / uploaded.name
            path.write_bytes(uploaded.getvalue())
            _queue('load', (path, not no_header, uploaded.name))

        with st.form('find', border=False):
            description = st.text_input('Or describe the table you need', disabled=busy or not datalake)
            if st.form_submit_button('Search the lake', disabled=busy or not datalake) and description.strip():
                _queue('find', description.strip())
        if not datalake:
            st.caption('Searching the lake needs the app started with --datalake <dir>.')

        if manager is not None and manager.table is not None:
            st.caption(f'{manager.table.name}: {len(manager.table.df)} rows x {len(manager.table.columns)} columns')
            st.dataframe(manager.table.df, height='stretch')


def _run(kind: str, payload: Any, config: str, datalake: Optional[str]) -> None:
    state = st.session_state
    manager: Optional[DialogueManager] = state['manager']
    try:
        if kind == 'load':
            path, header, name = payload
            state['manager'] = DialogueManager(manager.config if manager else AgentConfig.load(config), config,
                                               manager.models if manager else None)
            state['session'] += 1
            _add('response', response=state['manager'].handle(LoadTable(path, header=header, name=name)))
        elif kind == 'find':
            _add('user', text=f'Find a query table: {payload}')
            _add('found', found=candidates(payload, config, AgentConfig.load(config).k_default, datalake))
        elif kind == 'event':
            label, event = payload
            _add('user', text=label)
            _add('response', response=manager.handle(event))
        else:
            action, rank, columns, column = payload
            _add('user', text=f'{ACTED[action]} table {rank}')
            done = perform(manager, action, rank, datalake, columns, column)
            if isinstance(done, str):
                _add('note', text=done)
            else:
                state['manager'], text, response = done
                _add('note', text=text)
                state['session'] += 1
                _add('response', response=response)
    except Exception:
        if manager is not None:
            manager.record.write(event='error', step=kind, traceback=traceback.format_exc())
        _add('error', text=FAILED)


def _streamed(status, box) -> Listener:
    drawn = [0.0]

    def listener(name: str, index: int, thinking: List[str], text: List[str]) -> None:
        if index or time.perf_counter() - drawn[0] < REDRAW:
            return
        drawn[0] = time.perf_counter()
        status.update(label=f'{STEPS.get(name, name)}: {"writing" if text else "thinking"}')
        if thinking:
            with box.container():
                st.caption('The first sample\'s thinking, as it arrives:')
                st.text(''.join(thinking)[-THINKING_TAIL:])

    return listener


def _add(kind: str, **entry: Any) -> None:
    st.session_state['transcript'].append({'kind': kind, 'session': st.session_state['session'], **entry})


def main() -> None:
    arguments = _arguments()
    st.set_page_config(page_title='Blend Agent', layout='wide')
    st.markdown(PANE_CSS.replace('$BACKGROUND$', _background()), unsafe_allow_html=True)
    state = st.session_state
    for name, value in {'panel_open': True, 'manager': None, 'transcript': [], 'session': 0,
                        'round': 0, 'queued': None}.items():
        state.setdefault(name, value)
    queued = state['queued']
    busy = queued is not None
    manager: Optional[DialogueManager] = state['manager']

    with st.container(key='app_header'):
        title, new_chat, toggle = st.columns([4, 1, 1])
    title.subheader('Blend Agent')
    if new_chat.button('Start new chat', key='new_chat', disabled=busy):
        state.update(manager=None, transcript=[], upload_id=None)
        state['round'] += 1
        st.rerun()
    if toggle.button('Hide table' if state['panel_open'] else 'Show table', key='toggle_panel', disabled=busy):
        state['panel_open'] = not state['panel_open']
        st.rerun()

    if state['panel_open']:
        chat, panel = st.columns([2, 1])
        _fill_panel(panel, busy, arguments.datalake)
    else:
        chat = st.container()
    chat = chat.container(height=PANE_HEIGHT, key='chat_pane')

    message = st.chat_input(placeholder='Ask for tables' if manager else 'Load a query table first',
                            disabled=busy or manager is None or manager.table is None, key='chat_input')
    if message:
        _queue('event', (message, Message(message)))

    with chat:
        if manager is None:
            st.caption('Upload a query table, or describe one and search the lake for it.')
        _render(manager, queued, arguments.datalake)
        if busy:
            kind, payload = queued
            with st.status(SPINNER[kind], expanded=True) as status, listening(_streamed(status, st.empty())):
                _run(kind, payload, arguments.config, arguments.datalake)
            state['queued'] = None
            st.rerun()


main()
