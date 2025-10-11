PREFIX ?= /usr/local
MANDIR ?= $(PREFIX)/share/man
MAN1DIR := $(DESTDIR)$(MANDIR)/man1
MAN3DIR := $(DESTDIR)$(MANDIR)/man3
MAN5DIR := $(DESTDIR)$(MANDIR)/man5

MANPAGES := \
  man/neural_audio_tokenizer.1 \
  man/neural_audio_tokenizer.3 \
  man/lam_audio_tokens.5

INSTALL ?= install
MKDIR_P ?= $(INSTALL) -d
GZIP ?= gzip -n -9

.PHONY: all man install-man uninstall-man help

all: man

man:
	@echo "Man pages present:" $(MANPAGES)

install-man: $(MANPAGES)
	@$(MKDIR_P) $(MAN1DIR) $(MAN3DIR) $(MAN5DIR)
	@for f in $(MANPAGES); do \
	  sec=$${f##*.}; \
	  base=$$(basename $$f); \
	  case $$sec in \
	    1) out=$(MAN1DIR)/$$base.gz ;; \
	    3) out=$(MAN3DIR)/$$base.gz ;; \
	    5) out=$(MAN5DIR)/$$base.gz ;; \
	    *) echo "Unknown man section: $$f"; exit 1 ;; \
	  esac; \
	  $(GZIP) -c $$f > $$out; \
	  echo "installed $$out"; \
	done

uninstall-man:
	@rm -f $(MAN1DIR)/neural_audio_tokenizer.1.gz \
	       $(MAN3DIR)/neural_audio_tokenizer.3.gz \
	       $(MAN5DIR)/lam_audio_tokens.5.gz || true
	@echo "uninstalled man pages"

help:
	@echo "Targets:"
	@echo "  man            - show man page files"
	@echo "  install-man    - install compressed man pages to $(MANDIR)"
	@echo "  uninstall-man  - remove installed man pages"
	@echo "Variables: PREFIX=$(PREFIX) DESTDIR=$(DESTDIR)"

