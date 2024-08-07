#ifndef __qtrees_classifier__
#define __qtrees_classifier__

/******************************************************************************
 * INCLUDES
 ******************************************************************************/

#include "LuaObject.h"
#include "OsApi.h"
#include "bathy/BathyClassifier.h"
#include "bathy/BathyParms.h"

/******************************************************************************
 * BATHY CLASSIFIER
 ******************************************************************************/

class QtreesClassifier: public BathyClassifier
{
    public:

        /*--------------------------------------------------------------------
         * Constants
         *--------------------------------------------------------------------*/

        static const char* CLASSIFIER_NAME;
        static const char* QTREES_PARMS;
        static const char* DEFAULT_QTREES_MODEL;

        /*--------------------------------------------------------------------
         * Typedefs
         *--------------------------------------------------------------------*/

        struct parms_t {
            string model;       // filename for xgboost model
            bool set_class;     // whether to update class_ph in extent
            bool set_surface;   // whether to update surface_h in extent
            bool verbose;       // verbose settin gin XGBoost library
            parms_t(): 
                model (DEFAULT_QTREES_MODEL),
                set_class (true),
                set_surface (true),
                verbose (true) {};
            ~parms_t() {};
        };

        /*--------------------------------------------------------------------
         * Methods
         *--------------------------------------------------------------------*/

        static int  luaCreate   (lua_State* L);
        static void init        (void);

        bool run (const vector<BathyParms::extent_t*>& extents) override;

    protected:

        /*--------------------------------------------------------------------
         * Methods
         *--------------------------------------------------------------------*/

        QtreesClassifier (lua_State* L, int index);
        ~QtreesClassifier (void) override;

        /*--------------------------------------------------------------------
         * Data
         *--------------------------------------------------------------------*/

        parms_t parms;
};

#endif  /* __qtrees_classifier__ */
