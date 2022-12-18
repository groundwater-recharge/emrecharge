from collections import namedtuple
from traitlets import HasTraits, TraitType
from ipywidgets import Output

RegisteredWidget = namedtuple('RegisteredWidget', ['name','widget'])

class AppState(HasTraits):
    widgets: dict[str, any]
    
    @property
    def traits(self):
        return list(self.__dict__.keys())

    def to_dict(self):
        return self.__dict__
    
    def register_stateful_widget(self, widget: any, trait_name: str, traitlet: TraitType):
        if not hasattr(self, 'widgets'):
            self.widgets = {}
        self.add_traits(**{trait_name: traitlet})
        self.widgets[trait_name] = widget
        self.widgets[trait_name].observe(lambda _: self.update_from_ui(), names=["value"])
    
    def register_stateful_property(self, trait_name: str, traitlet: TraitType):
        self.add_traits(**{trait_name: traitlet})
    
    def register_widget_observer(self, observes: str, observer):
        self.widgets[observes].observe(observer, names=["value"])
        
    def update_from_ui(self):
        for name in self.widgets.keys():
            setattr(self, name, self.widgets[name].value)

    def restore_ui(self):
        for name in self.widgets.keys():
            self.widgets[name].value = self[name]
    
    def observe_trait(self, trait_name: str, fn):
        def observer(state):
            if state["name"] == trait_name:
                fn(state)
        
        return self.observe(observer)
        
    
    @property
    def outlet(self):
        out = Output()
        def print_state(s):
            with out:
                print(s)
        self.observe(print_state)
        with out:
            print("AppState Outlet:")
        return out
