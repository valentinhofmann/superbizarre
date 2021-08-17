class Derivator:

    def __init__(self, f_pfxes='pfxes.txt', f_sfxes='sfxes.txt', f_stems='stems.txt', n_stems=None):

        self.pfxes = list()
        self.sfxes = list()
        self.stems = set()

        # Load prefixes
        with open(f_pfxes, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                self.pfxes.append(l.strip().lower())

        # Load suffixes
        with open(f_sfxes, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                self.sfxes.append(l.strip().lower())

        # Load stems
        with open(f_stems, 'r') as f:
            for l in f:
                if l.strip() == '':
                    continue
                self.stems.add(l.strip().lower())
                if n_stems and len(self.stems) == n_stems:
                    break

    def segment(self, form):

        found_pfxes = []
        pfx_happy = True

        # Outer loop to check prefixes
        while pfx_happy:

            found_sfxes = []
            form_temp = form
            sfx_happy = True

            # Inner loop to check suffixes
            while sfx_happy:

                if form_temp == '':
                    break

                if form_temp in self.stems:

                    return found_pfxes[:], form_temp, found_sfxes[::-1]

                elif len(form_temp) > 2 and is_cons(form_temp[-1]) and len(found_sfxes) > 0 \
                        and is_vowel(found_sfx[0]) and form_temp + 'e' in self.stems:

                    form_temp = form_temp + 'e'

                    return found_pfxes[:], form_temp, found_sfxes[::-1]

                elif len(form_temp) > 2 and is_cons(form_temp[-1]) and form_temp[-1] == form_temp[-2] \
                        and is_vowel(form_temp[-3]) and len(found_sfxes) > 0 and form_temp[:-1] in self.stems:

                    form_temp = form_temp[:-1]

                    return found_pfxes[:], form_temp, found_sfxes[::-1]

                # Need to find suffix to stay in inner loop
                sfx_happy = False
                found_sfx = ''
                for sfx in self.sfxes:
                    if form_temp.endswith(sfx):
                        sfx_happy = True
                        if sfx == 'ise':
                            found_sfx = 'ize'
                        else:
                            found_sfx = sfx
                        found_sfxes.append(found_sfx)
                        form_temp = form_temp[:-len(sfx)]
                        break

                # Check for special phonological alternations
                try:
                    if found_sfx in {'ation', 'ate'} and form_temp[-4:] == 'ific':
                        form_temp = form_temp[:-4] + 'ify'
                    elif found_sfx == 'ness' and form_temp[-1] == 'i':
                        form_temp = form_temp[:-1] + 'y'
                    elif form_temp[-4:] == 'abil':
                        form_temp = form_temp[:-4] + 'able'
                except IndexError:
                    continue

            # Need to find prefix to stay in outer loop
            pfx_happy = False
            for pfx in self.pfxes:
                if form.startswith(pfx):
                    # Check addition of false prefixes
                    pfx_happy = True
                    if pfx in {'im', 'il', 'ir'}:
                        found_prefix = 'in'
                    else:
                        found_prefix = pfx
                    found_pfxes.append(found_prefix)
                    form = form[len(pfx):]
                    break

        return ''

    def derive(self, form, mode='bundles'):

        try:
            pfxes, root, sfxes = self.segment(form)
            if mode == 'roots':
                return root
            if mode == 'bundles':
                return ''.join(p + '_' for p in pfxes), root, ''.join('_' + s for s in sfxes)
            if mode == 'morphemes':
                return pfxes, root, sfxes
        except ValueError:
            if mode == 'roots':
                return form
            if mode == 'bundles':
                return '', form, ''
            if mode == 'morphemes':
                return [], form, []

    def tokenize(self, form_list, mode='bundles'):

        if isinstance(form_list, str):

            form_list = form_list.strip().split()

        output = list()

        for f in form_list:

            d = self.derive(f, mode)

            if mode == 'roots':
                output.append(d)
            if mode == 'bundles':
                output.extend([s for s in d if s != ''])
            if mode == 'morphemes':
                if len(d[0]) > 0:
                    output.extend(d[0])
                output.append(d[1])
                if len(d[2]) > 0:
                    output.extend(d[2])

        return output


# Define function to detect vowels
def is_vowel(char):
    return char.lower() in 'aeiou'


# Define function to detect consonants
def is_cons(char):
    return char.lower() in 'bdgptkmnlrszfv'
