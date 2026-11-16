import { Search } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Input } from '@/components/ui/input'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { JOB_FILTERS, type JobFilterId } from './api'

export interface JobFiltersProps {
  filter: JobFilterId
  onFilterChange: (filter: JobFilterId) => void
  search: string
  onSearchChange: (search: string) => void
}

/**
 * UX §6 "Úlohy": server-side state tabs plus a client-side search over
 * already-loaded rows only (design §3.4 -- there is no server-side label
 * search in v1).
 */
export function JobFilters({ filter, onFilterChange, search, onSearchChange }: JobFiltersProps) {
  const { t } = useTranslation()

  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <Tabs
        value={filter}
        onValueChange={(value) => {
          onFilterChange(value as JobFilterId)
        }}
      >
        <TabsList>
          {JOB_FILTERS.map((id) => (
            <TabsTrigger key={id} value={id}>
              {t(`jobs.filter.${id}`)}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      <div className="relative sm:w-64">
        <Search
          aria-hidden="true"
          className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          type="search"
          value={search}
          onChange={(event) => {
            onSearchChange(event.target.value)
          }}
          placeholder={t('jobs.search')}
          aria-label={t('jobs.search')}
          className="pl-8"
        />
      </div>
    </div>
  )
}
