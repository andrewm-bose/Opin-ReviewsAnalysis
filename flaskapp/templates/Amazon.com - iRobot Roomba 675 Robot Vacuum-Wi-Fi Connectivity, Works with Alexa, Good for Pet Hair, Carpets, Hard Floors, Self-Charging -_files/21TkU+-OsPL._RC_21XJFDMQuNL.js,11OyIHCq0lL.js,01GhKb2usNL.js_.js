(function(e){var d=window.AmazonUIPageJS||window.P,g=d._namespace||d.attributeErrors,a=g?g("TwisterHardlinesDetailPageAsset"):d;a.guardFatal?a.guardFatal(e)(a,window):a.execute(function(){e(a,window)})})(function(e,d,g){e.when("A","jQuery","atf").execute(function(a,b){a.on.ready(function(){function c(a,c,d,e,f){b(a).slideToggle(300);b(c).slideToggle(300);b(d).toggleClass("tmmShowPrompt tmmHidePrompt");b(e).toggleClass("tmmShowPrompt tmmHidePrompt");b(f).toggleClass("rotate")}function f(a){a.position();
a.width();b(".swatchElement");a.find(".swatchElement").each(function(){b(this).attr("data-width",b(this).width())});b("#formats").find(".a-row").removeClass("nonJSFormats")}function g(){var a=b("#formats"),c=a.width()-40,d=0;a.find(".swatchElement").each(function(){b(this).is(":visible")&&(d+=b(this).width())});d>c?b(a.find(".swatchElement:visible").get().reverse()).each(function(){var a=0,e=!1;b(this).find(".format").hasClass("a-button-selected")?(a=b(this).prev(".swatchElement").width(),e=b(this).prev(".swatchElement").is(":hidden"),
b(this).prev(".swatchElement").hide()):(a=b(this).width(),e=b(this).is(":hidden"),b(this).hide());e||(d-=a);return d>c}):d<c&&a.find(".swatchElement").each(function(){if(b(this).is(":hidden"))return b(this).attr("data-width")<=c-d&&b(this).show(),!1})}var p=0,h=0,l,m=function(a){"undefined"!=typeof a&&(l=new a.ImpressionLogger("dpbxapps","bxapps-atfMarker",!0,!0))};"undefined"!=typeof amznJQ?amznJQ.available("DPClientLogger",function(){m(d.DPClientLogger)}):e.when("DPClientLogger").execute(m);a.declarative("tmm-see-more-editions-click",
["click"],function(e){var f=e.data,g=f.metabindingUrl;if(e.$target.hasClass("a-link-expander")||e.$target.parent().hasClass("a-link-expander")){var f=f.metabindingPlaceHolder,h="#metabinding_row_top_"+f,k="#metabinding_row_bottom_"+f,l="#editionsSeePrompt_"+f,m="#editionsHidePrompt_"+f,n="#editionsIcon_"+f,p="isAjaxComplete_"+f,q="isAjaxInProgress_"+f;e="#tmmSpinnerDiv_"+f;var t=a.state("mediamatrix-state"),f=t["url_"+f].replace(/&amp;/g,"&");b("#formats .tmmErrorClass").hide();"1"===t[p]||"1"===
t[q]?c(h,k,l,m,n):(a.ajax(f,{method:"get",success:function(){t[p]="1";a.state("mediamatrix-state",t);c(h,k,l,m,n)},indicator:b(e),error:function(){b($tmmErrorDiv).show();t[q]="0";a.state("mediamatrix-state",t)}}),t[q]="1",a.state("mediamatrix-state",t))}else"#"!==g&&(d.location=g)});if(0<b("#formats > .a-link-expander").length){var n=d.ue;n&&n.count&&n.count("mediaMatrixExpanderPresent",1)}b("#formats > .a-link-expander").click(function(){c("#tmmSwatches","#twister","#showMoreFormatsPrompt","#hideMoreFormatsPrompt",
"#formatsIcon");if(!h){var a=d.ue;a&&a.count&&a.count("mediaMatrixExpanderClicked",1);h=1}l&&b("#twister").is(":visible")&&!p&&(l.logImpression("tmm-show-more-formats-viewed"),p=1)});var n=a.state("mediamatrix-state"),t="1";"undefined"!==typeof n&&"undefined"!==typeof n.showHybridMediaMatrix&&"1"===n.showHybridMediaMatrix?t="0":"undefined"!==typeof n&&"undefined"!==typeof n.isDvdWeblabEnabled&&"1"===n.isDvdWeblabEnabled&&(t="0");"1"===t&&(b("#formats.responsive").each(function(){f(b(this))}),g(),
a.on("resize",g));b("#formats .unselected .format").mouseenter(function(){b(this).find(".a-color-secondary").addClass("a-color-price").removeClass("a-color-secondary")}).mouseleave(function(){b(this).find(".a-color-price").addClass("a-color-secondary").removeClass("a-color-price")});b("#landingItemRentalLink").attr("href","javascript:document.getElementById('rentBuySection').click();")})});e.when("A","jQuery","a-dropdown").execute(function(a,b,c){function e(a){return a?a.replace(/&#0/g,"&#"):undef}
function g(a){for(var b=a.getOptions(),b=b?b.size():0,c=1;c<b;c++)a.removeOption(1)}function p(a,c){var d=a.getOption(0),g=d.info()[0];d.remove();b.each(c,function(b,c){a.addOption({text:u[e(c)],value:c,css_class:"dropdownAvailable"},1)});a.addOption(g,0);a.val(g.value)}function h(a){B=a.value;g(C);"-1"!==B&&(l(),p(C,Object.keys(t[e(B)])))}function l(){Object.keys||(Object.keys=function(){var a=Object.prototype.hasOwnProperty,b=!{toString:null}.propertyIsEnumerable("toString"),c="toString toLocaleString valueOf hasOwnProperty isPrototypeOf propertyIsEnumerable constructor".split(" "),
d=c.length;return function(e){if("object"!==typeof e&&("function"!==typeof e||null===e))throw new TypeError("Object.keys called on non-object");var f=[],g;for(g in e)a.call(e,g)&&f.push(g);if(b)for(g=0;g<d;g++)a.call(e,c[g])&&f.push(c[g]);return f}}())}function m(a){z=a.value;"-1"!==z&&"-1"!==B&&(d.location=v[t[e(B)][e(z)]])}function n(a){a=a.value;"-1"!==a&&(d.location=v[e(a)])}var t,w=a.state("dcdMetaData");if(w){var v=w.asinToDetailPageURLMap,u=w.truncatedValuesMap,x=w.dimensionKeys,A=x[0],B=-1,
z=-1,C;2===x.length?(t=w.doubleValuesToAsinMap,w=x[1],x=c.getSelect("native_dcd_dropdown_"+A),C=c.getSelect("native_dcd_dropdown_"+w),x.setValue("-1"),C.setValue("-1"),g(C),a.on("a:dropdown:selected:"+A,h),a.on("a:dropdown:selected:"+w,m)):(x=c.getSelect("native_dcd_dropdown_"+A),(c=w.selectedAsin)?x.setValue(c):x.setValue("-1"),a.on("a:dropdown:selected:asin-redirect",n))}})});
/* ******** */
(function(g){var k=window.AmazonUIPageJS||window.P,h=k._namespace||k.attributeErrors,e=h?h("PInfoHardlinesDetailPageAsset",""):k;e.guardFatal?e.guardFatal(g)(e,window):e.execute(function(){g(e,window)})})(function(g,k,h){g.when("A","jQuery","atf").execute(function(e,a,g){function h(){var d=parseInt(a("#byline").width()),b=0;a("#byline .author").each(function(){a(this).is(".notFaded")&&(b+=a(this).outerWidth())});b+=a("#byline .more").outerWidth();if(b>d){var e=a("#byline .author.notFaded").length;
a(a("#byline .author.notFaded").get().reverse()).each(function(c){b>d&&c<e-1&&(c=a(this).width(),a(this).removeClass("notFaded"),a(this).fadeOut(10,function(){a(this).hide()}),b-=c,a("#byline .moreCount").html(a("#byline .author").length-a("#byline .author.notFaded").length),a("#byline .more").addClass("notFaded").fadeIn(10))})}else if(b<d&&a("#byline .author.notFaded").length<a("#byline .author").length){var f=d-b;a("#byline .author").each(function(){if(!a(this).hasClass("notFaded")){var c=a(this).outerWidth();
c<=f?(f-=c,a(this).addClass("notFaded").fadeIn(10),c=a("#byline .author").length-a("#byline .author.notFaded").length,0===c?a("#byline .more").removeClass("notFaded").fadeOut(10):(a("#byline .moreCount").html(c),a("#byline .more").addClass("notFaded").fadeIn(10))):f=0}})}}a("#byline .showMoreLink").click(function(){a("#byline .author").each(function(){a(this).hasClass("notFaded")||a(this).addClass("notFaded").fadeIn(0);a("#byline .more").removeClass("notFaded").fadeOut(0)});return!1});a("#byline .contributorNameID").mouseenter(function(){var d=
a(this).attr("data-asin"),b={},l="isAjaxComplete_"+d,f="isAjaxInProgress_"+d,c=e.state("popoverImage-state");b.entityID=d;"1"!==c[l]&&"1"!==c[f]&&(a.ajax({url:"/gp/product/utility/by-line/book-contributor-details/ajax/author-image.html",data:b,dataType:"html",timeout:1E3,success:function(b){a("#contributorImageContainer"+d).get(0).innerHTML=b;c[l]="1";e.state("popoverImage-state",c)},error:function(){c[f]="0";e.state("popoverImage-state",c)}}),c[f]="1",e.state("popoverImage-state",c))});(function(d){var b=
parseInt(d.width()),e=parseInt(d.find(".more").outerWidth()),f=0;a("#byline .author").each(function(){a(this).is(".notFaded")&&(f+=a(this).outerWidth())});a("#byline .more").attr("data-width",e);var c=b-e;if(c>f)d.find(".author").each(function(){var b=a(this).index();3<=parseInt(b)?(b=a(this).outerWidth(),a(this).attr("data-width",b),b<c?(c-=b,a(this).addClass("notFaded"),a(this).fadeIn("slow")):(c=0,a(this).fadeOut("slow"))):(b=a(this).outerWidth(),a(this).attr("data-width",b),c-=b)});else{var g=
a("#byline .author.notFaded").length;a(a("#byline .author.notFaded").get().reverse()).each(function(b){c<f&&b<g-1&&(b=a(this).outerWidth(),f-=b,a(this).fadeOut("slow").removeClass("notFaded"))})}b=d.find(".author").length-d.find(".author.notFaded").length;0<b&&(d.find(".moreCount").html(b),d.find(".more").fadeIn("slow"),d.find(".more").addClass("notFaded"))})(e.$("#byline"));a(k).resize(function(){h()})});g.when("A","jQuery","atf").execute(function(e,a,g){function h(){var d=parseInt(a("#bylineInfo").width()),
b=0;a("#bylineInfo .author").each(function(){a(this).is(".notFaded")&&(b+=a(this).outerWidth())});b+=a("#bylineInfo .more").outerWidth();if(b>d){var e=a("#bylineInfo .author.notFaded").length;a(a("#bylineInfo .author.notFaded").get().reverse()).each(function(c){b>d&&c<e-1&&(c=a(this).width(),a(this).removeClass("notFaded"),a(this).fadeOut(10,function(){a(this).hide()}),b-=c,a("#bylineInfo .moreCount").html(a("#bylineInfo .author").length-a("#bylineInfo .author.notFaded").length),a("#bylineInfo .more").addClass("notFaded").fadeIn(10))})}else if(b<
d&&a("#bylineInfo .author.notFaded").length<a("#bylineInfo .author").length){var f=d-b;a("#bylineInfo .author").each(function(){if(!a(this).hasClass("notFaded")){var c=a(this).outerWidth();c<=f?(f-=c,a(this).addClass("notFaded").fadeIn(10),c=a("#bylineInfo .author").length-a("#bylineInfo .author.notFaded").length,0===c?a("#bylineInfo .more").removeClass("notFaded").fadeOut(10):(a("#bylineInfo .moreCount").html(c),a("#bylineInfo .more").addClass("notFaded").fadeIn(10))):f=0}})}}a("#bylineInfo .showMoreLink").click(function(){a("#bylineInfo .author").each(function(){a(this).hasClass("notFaded")||
a(this).addClass("notFaded").fadeIn(0);a("#bylineInfo .more").removeClass("notFaded").fadeOut(0)});return!1});a("#bylineInfo .contributorNameID").mouseenter(function(){var d=a(this).attr("data-asin"),b={},g="isAjaxComplete_"+d,f="isAjaxInProgress_"+d,c=e.state("popoverImage-state");b.entityID=d;"1"!==c[g]&&"1"!==c[f]&&(a.ajax({url:"/gp/product/utility/by-line/book-contributor-details/ajax/author-image.html",data:b,dataType:"html",timeout:1E3,success:function(b){a("#contributorImageContainer"+d).get(0).innerHTML=
b;c[g]="1";e.state("popoverImage-state",c)},error:function(){c[f]="0";e.state("popoverImage-state",c)}}),c[f]="1",e.state("popoverImage-state",c))});(function(d){var b=parseInt(d.width()),e=parseInt(d.find(".more").outerWidth()),f=0;a("#bylineInfo .author").each(function(){a(this).is(".notFaded")&&(f+=a(this).outerWidth())});a("#bylineInfo .more").attr("data-width",e);var c=b-e;if(c>f)d.find(".author").each(function(){var b=a(this).index();3<=parseInt(b)?(b=a(this).outerWidth(),a(this).attr("data-width",
b),b<c?(c-=b,a(this).addClass("notFaded"),a(this).fadeIn("slow")):(c=0,a(this).fadeOut("slow"))):(b=a(this).outerWidth(),a(this).attr("data-width",b),c-=b)});else{var g=a("#bylineInfo .author.notFaded").length;a(a("#bylineInfo .author.notFaded").get().reverse()).each(function(b){c<f&&b<g-1&&(b=a(this).outerWidth(),f-=b,a(this).fadeOut("slow").removeClass("notFaded"))})}b=d.find(".author").length-d.find(".author.notFaded").length;0<b&&(d.find(".moreCount").html(b),d.find(".more").fadeIn("slow"),d.find(".more").addClass("notFaded"))})(e.$("#bylineInfo"));
a(k).resize(function(){h()})});g.when("A","jQuery").register("product-description-fix",function(e,a){return{fixTableIssue:function(){a("#productDescription .productDescriptionWrapper table").each(function(){var e=a(this).attr("width");"undefined"!==typeof e?a(this).css("width",e):a(this).css("width","auto")})}}})});
/* ******** */
(function(d){var e=window.AmazonUIPageJS||window.P,h=e._namespace||e.attributeErrors,f=h?h("OffersHardlinesDetailPageAsset",""):e;f.guardFatal?f.guardFatal(d)(f,window):f.execute(function(){d(f,window)})})(function(d,e,h){d.when("A","jQuery","atf").register("accordionBuyBoxJS",function(f,a,h){var g={updateCssClass:function(b){var c=a("#rbbContainer");c.find(".selected .a-icon-radio-active").removeClass("a-icon-radio-active").addClass("a-icon-radio-inactive");c.find(".selected .offer-price").removeClass("a-color-price").addClass("a-color-secondary");
b.parents(".rbbSection").removeClass("unselected").addClass("selected");b.parents(".rbbSection").toggleClass("dp-accordion-active",500);b.find(".a-icon-radio-inactive").removeClass("a-icon-radio-inactive").addClass("a-icon-radio-active");b.find(".offer-price").removeClass("a-color-secondary").addClass("a-color-price");a("#rbbContainer .rbbSection .rbbContent").find(".offer-price").removeClass("a-color-secondary").addClass("a-color-price")},animate:function(b){var c=a("#rbbContainer"),f=b.attr("id"),
d,e;c.find(".rbbSection").each(function(b,c){a(c).find(".rbbHeader")[0].id==f?d=a(c):a(c).hasClass("selected")&&(e=a(c))});e.find(".rbbContent").slideUp(500,function(){e.removeClass("selected").addClass("unselected");e.toggleClass("dp-accordion-active",500)});d.find(".rbbContent").slideDown(500);a.browser.msie&&7==parseInt(a.browser.version,10)&&(d.find(".rbbContent").css("display","inline"),setTimeout(function(){d.find(".rbbContent").css("display","block")},505))},oneClickJS:function(){a("#one-click-button, #one-click-button-ubb").click(function(){var b=
"https://"+e.location.host+a("#addToCart").attr("action");a("#addToCart").attr("action",b);return!0})},usedBuyBoxJS:function(){a("#one-click-button-ubb, #usedbuyBox #add-to-cart-button-ubb").click(function(){var b=a("#addToCart"),c=b.attr("action").replace("ref\x3ddp_start-bbf_1_glance","ref\x3ddp_start-ubbf_1_glance");b.attr("action",c);return!0})},initialize:function(){a(".rbbHeaderLink").attr("href","javascript:void(0);");a("#rbbContainer .rbbSection.unselected .dp-accordion-inner").hide();a("#rbbContainer .rbbSection .rbbHeader").click(function(b){a(b.target);
b=a(this);var c=b.hasClass("rbbHeader")?b:b.parents(".rbbHeader");if(c.parents(".rbbSection").hasClass("selected"))return!1;b=a("#rbbContainer .rbbSection.selected .rbbHeader").attr("id");var d=c.attr("id");g.updateCssClass(c);a("#abbWrapper [id^\x3dmbb-offeringID-]").each(function(){this.checked=!1});c=a(this);g.animate(c);f.trigger("buybox:accordion:changed",d,b);return!0});g.oneClickJS();g.usedBuyBoxJS()}};d.when("a-popover").execute("used-buy-box-condition-popover",function(b){var c=a("#usedItemConditionInfoLink"),
d=b.create(c,{width:250,position:"triggerBottom",activate:f.capabilities.touch?"onmousemove":"onclick",closeButton:!1,popoverLabel:"Used condition details",name:"usedItemConditionDetailsPopover",dataStrategy:"preload"});c.mouseleave(function(){d.hide()})});return g})});
/* ******** */
(function(a){var c=window.AmazonUIPageJS||window.P,d=c._namespace||c.attributeErrors,b=d?d("DetailPagePromotionalBundleAssets",""):c;b.guardFatal?b.guardFatal(a)(b,window):b.execute(function(){a(b,window)})})(function(a,c,d){a.when("A","ready").execute("kbpMobileExpander",function(b){var a=b.$;a(".pb-bundle-section-expander").click(function(b){a(".pb-bundle-mobile-box").removeClass("aok-hidden");a(".pb-bundle-expander-box").addClass("aok-hidden")})})});
/* ******** */
